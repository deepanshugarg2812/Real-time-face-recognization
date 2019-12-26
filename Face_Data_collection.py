# 1->In this we are going to capture the face and store them in the array form with extension .npy
# 2->Access the webcam then use cascade classifier object and detect multi scale method of it to wxtract face from image
# 3->Draw the rectangle arounf the face in the frame and use imshow method to show the frame
# 4->Define the data path to save the data
# 5->Define skip variable to save every 10th frame so use less space
# 6->Flatten the array in single row 
import cv2
import numpy as np

#Capture the video webcam
cap = cv2.VideoCapture(0)
#Make object of cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

face_data = [] #To save the locations
skip = 0
DataPath = './data/'

file_name = input('Enter the register no. of the person')
#Run a while loop until user presses q to stop getting frames
while True:
	ret,Frame = cap.read()
	#Convert the frame in the gray scale to reduce the complexity currently-:BGR
	gray_frame = cv2.cvtColor(Frame,cv2.COLOR_BGR2GRAY)

	#Detect the faces using detect multi scale method
	face_position = face_cascade.detectMultiScale(gray_frame,1.3,5)
	#Sort the positions on the basis of size that is w*h if there are more than ones frame
	faces = sorted(face_position,key=lambda f:f[2]*f[3],reverse=True)

	for face in faces:
		x,y,w,h = face
		cv2.rectangle(Frame,(x,y),(x+w,y+h),(255,0,0),2)
		#print(face) undo comment to check if frames are being read correctly
		#We need to save some extra offet of face in order to not to miss something
		offset = 10
		face_offset = gray_frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_offset = cv2.resize(face_offset,(100,100))

		cv2.imshow('Small Frame',face_offset)
		#print(face_offset) To check if slicing working correctly
		skip += 1
		if skip%10==0:
			face_data.append(face_offset)
			print(len(face_data))

	cv2.imshow('Video',Frame)
	#Method to exit the loop
	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed==ord('q'):
		break

face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

#Save the data to file
np.save(DataPath+file_name+'.npy',face_data)
print('Data saved successfully')

cap.release()
cv2.destroyAllWindows()