import os
import cv2
import numpy as np
from pathlib import Path
import torch
from torch.backends import cudnn
from nvidia.dali import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator 
import nvidia.dali.fn as fn
import nvidia.dali.types as types


class MegaagePreprocessor:
    def __init__(self, src_dir="megaage", dst_dir="megaage_processed"):
        # 启用硬件加速
        cudnn.benchmark = True
        torch.cuda.set_per_process_memory_fraction(0.9)  # 保留10%显存
        
        self.src_dir = src_dir  # 新增实例变量
        self.dst_dir = dst_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self._create_file_list()
        self.face_detector = self._init_detector()
        self._create_dirs(dst_dir)
        
    def _create_file_list(self):
        # 在生成 file_list.txt 时，同时保存文件名到一个列表
        self.file_names = []
        with open("file_list.txt", "w") as f:
            for subset in ["train", "test"]:
                subset_dir = Path(self.src_dir) / subset
                # 添加调试信息
                print(f"正在扫描目录: {subset_dir}")
                file_count = 0
                
                for img_file in subset_dir.glob("*.[jJ][pP][gG]"):
                    file_path = f"{subset}/{img_file.name}"
                    f.write(f"{file_path} 0\n")
                    self.file_names.append(file_path)
                    file_count += 1
                
                for img_file in subset_dir.glob("*.[pP][nN][gG]"):
                    file_path = f"{subset}/{img_file.name}"
                    f.write(f"{file_path} 0\n") 
                    self.file_names.append(file_path)
                    file_count += 1
                
                print(f"{subset} 子集找到 {file_count} 个文件")

        # 添加最终检查
        print(f"总文件数: {len(self.file_names)}")
        with open("file_list.txt", "r") as f:
            print(f"file_list.txt 行数: {len(f.readlines())}")

    def _init_detector(self):
        """加载YuNet人脸检测器"""
        model_path = Path(__file__).parent / "face_detection_yunet_2023mar.onnx"  # 添加模型路径处理
        if not model_path.exists():
            raise FileNotFoundError(f"找不到ONNX模型文件: {model_path}")
        
        # 使用YuNet专用初始化方法
        detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (320, 320),
            score_threshold=0.8,
            backend_id=cv2.dnn.DNN_BACKEND_CUDA,
            target_id=cv2.dnn.DNN_TARGET_CUDA
        )
        return detector

    def _create_dirs(self, dst_root):
        """创建多级存储目录"""
        Path(dst_root).mkdir(exist_ok=True)
        for subset in ["train", "test"]:
            (Path(dst_root)/subset).mkdir(exist_ok=True)

    def _gpu_pipeline(self):
        """构建DALI加速流水线"""
        pipe = Pipeline(batch_size=2048, num_threads=32, device_id=0, prefetch_queue_depth=4)
        
        with pipe:
            jpegs, labels = fn.readers.file(
                file_root=self.src_dir,
                random_shuffle=True,
                prefetch_queue_depth=4,
                name="Reader",
                file_list="file_list.txt"
            )
            images = fn.decoders.image(
                jpegs,
                device="mixed",
                output_type=types.RGB,
                hw_decoder_load=0.8  # 硬件解码负载优化
            )
            images = fn.resize(
                images,
                size=[512, 512],  # 使用size参数确保统一尺寸
                mode='not_smaller',
                interp_type=types.INTERP_LANCZOS3  # 高质量插值
            )
            images = fn.crop(images, crop=[512, 512])  # 确保最终尺寸完全一致
            pipe.set_outputs(images, labels)  # 同时输出图像和标签
        return pipe

    def _detect_and_crop(self, img_batch):
        """批量GPU人脸检测与裁剪"""
        cropped_faces = []
        for i in range(img_batch.shape[0]):
            img = img_batch[i].transpose(1, 2, 0).astype('uint8')
            
            # 设置输入尺寸并检测
            self.face_detector.setInputSize((img.shape[1], img.shape[0]))
            _, faces = self.face_detector.detect(img)
            
            if faces is None: 
                continue
                
            # 取第一个检测结果（已按置信度排序）
            x1, y1, w, h = map(int, faces[0][:4])
            cropped = img_batch[i, :, y1:y1+h, x1:x1+w]
            cropped_faces.append((cropped, (x1, y1, x1+w, y1+h)))
        
        return cropped_faces

    def process_dataset(self):
        """主处理流程"""
        pipe = self._gpu_pipeline()
        loader = DALIClassificationIterator(pipe, reader_name="Reader")
        
        # 关闭随机打乱以确保顺序一致（如果需要）
        file_index = 0
        total_processed = 0

        for batch_idx, data in enumerate(loader):
            img_batch = data[0]["data"].contiguous().cpu().numpy()
            
            for idx, img in enumerate(img_batch):
                if file_index >= len(self.file_names):
                    print(f"警告: 已处理 {total_processed} 个文件，但DALI仍在生成数据")
                    print("可能 file_list.txt 与实际文件不匹配")
                    return

                # 根据 file_index 获取原始文件名
                original_file = self.file_names[file_index]
                file_index += 1
                total_processed += 1
                dst_path = Path(self.dst_dir) / Path(original_file)
                dst_path.parent.mkdir(parents=True, exist_ok=True)

                 # 添加调试信息
                print(f"处理文件: {original_file}, 原始形状: {img.shape}")
                
                # 修正维度转换逻辑
                if img.shape == (3, 512, 512):  # CHW格式
                    img = img.transpose(1, 2, 0)  # 转为HWC
                elif img.shape == (512, 512, 3):  # 已经是HWC格式
                    pass  # 无需转换
                else:
                    print(f"警告: 非常规图像形状 {img.shape}")
                    img = img.reshape(512, 512, 3)  # 强制重塑形状
                
                img = img.astype('uint8')
                print(f"保存前图像形状: {img.shape}")
                
                if not cv2.imwrite(str(dst_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    print(f"错误: 无法保存图像 {dst_path}")
        
        print(f"处理完成，共处理 {total_processed}/{len(self.file_names)} 个文件")


                    
    def _save_results(self, batch, paths):
        """显存直写存储"""
        for (face, bbox), path in zip(batch, paths):
            # 使用实例变量构建路径
            dst_path = Path(self.dst_dir) / Path(path).parent.name / Path(path).name
            # 转换回CPU保存
            cv2.imwrite(str(dst_path), face.transpose(1,2,0).astype('uint8'), 
                      [cv2.IMWRITE_JPEG_QUALITY, 95])

if __name__ == "__main__":
    preprocessor = MegaagePreprocessor()
    preprocessor.process_dataset()
