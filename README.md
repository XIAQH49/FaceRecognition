# FaceRecognition
Gender and age recognition through facial features.

## Project Overview
This project aims to build a face gender and age recognition system based on the Qt framework. The system will combine computer vision and machine learning technologies to achieve efficient processing and analysis of facial images and accurately identify the gender and age information of the person in the image. Users can interact with the system through a simple and intuitive graphical user interface (GUI), select local pictures or open the camera to capture pictures in real time, and the system will quickly give and display the recognition results.

## Project Goals
### Functionality
1. **Image selection and display:** Support users to select facial image files from local, and the selected images will be clearly displayed in the system interface, and support multiple common image formats, such as PNG, JPEG, etc.
2. **Real-time camera recognition:** Users can turn on the camera, and the system will process the images captured by the camera in real time, identify the gender and age of the faces in them, and display the recognition results in real time.
3. **Recognition result presentation:** For each face image or face in the real-time picture, the system will accurately identify its gender (male or female) and approximate age range, and present the results in an intuitive way on the interface, such as text annotation on the face image.
4. **User-friendly interaction:** The simple and easy-to-use Qt interface is designed so that users can operate easily, even non-professionals can quickly get started. At the same time, necessary prompt information is provided during the operation to enhance the user experience.
### Performance indicators
1. **Recognition accuracy:** On common face image datasets, the gender recognition accuracy is over 90%, and the age recognition error is controlled within ±5 years.
2. **Processing speed:** For a single face image, the recognition time is controlled within 1 second; for real-time camera images, maintain a smooth recognition frame rate of no less than 10 frames/second.
### System stability
Ensure that the system remains stable during long-term operation without crashing or abnormal exit. For abnormal input (such as non-face images, damaged image files, etc.), the system can give clear error prompts to avoid program crashes.

## Project Structure
>.gitignore  
>AgeGender.py  
>README.md  
>main_window.py  
>main_window.ui  
>text.py  
>models/  
>+ age_deploy.prototxt  
>+ age_net.caffemodel  
>+ gender_deploy.prototxt  
>+ gender_net.caffemodel  
>+ opencv_face_detector.pbtxt  
>+ opencv_face_detector_uint8.pb


  
