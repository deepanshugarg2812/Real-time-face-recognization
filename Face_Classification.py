# Steps to follow to classifiy faces
# 1->Import importnant libraries
# 2->Import the data sets saved from the face data classification and apply labels to the data
# 3->Convert data set into single by using the concatenate function
# 4->Name it like trainset
# 5->Capture the frames as done in the data collection and pass the dat after flattening to the knn algorithm 
# 6->Classfy the name by getting the name from the name dictonary
# 7->Make the rectangular box anf use pyt text function to add name on top of the box
import cv2
import numpy as np
import os

########## KNN CODE ############
def distance(v1, v2):
	# Eucledian 
	return np.sqrt(((v1-v2)**2).sum())

def knn(train, test, k=5):
	dist = []
	
	for i in range(train.shape[0]):
		# Get the vector and label
		ix = train[i, :-1]
		iy = train[i, -1]
		# Compute the distance from test point
		d = distance(test, ix)
		dist.append([d, iy])
	# Sort based on distance and get top k
	dk = sorted(dist, key=lambda x: x[0])[:k]
	# Retrieve only the labels
	labels = np.array(dk)[:, -1]
	
	# Get frequencies of each label
	output = np.unique(labels, return_counts=True)
	# Find max frequency and corresponding label
	index = np.argmax(output[1])
	return output[0][index]
################################

skip = 0
DataPath = './data/'

face_data = []
name = {} #Map name with the id it belongs
labels = [] #Apply labels for the given file
class_id = 0 #Class id increment by one on each file

for fx in os.listdir(DataPath):
	if fx.endswith('.npy'):
		name[class_id] = fx[:-4]
		print('Loaded'+fx)
		data_item = np.load(DataPath+fx)
		face_data.append(data_item)
		labels.append(class_id * np.ones((data_item.shape[0],)))
		class_id += 1

#Making the data for training like X_train,Y_train
face_data = np.concatenate(face_data,axis=0)
labels = np.concatenate(labels,axis=0).reshape((-1,1))

print(face_data.shape)
print(labels.shape)

trainset = np.concatenate((face_data,labels),axis=1)
print(trainset.shape)

#Testing part

#Capture the video webcam
cap = cv2.VideoCapture(0)
#Make object of cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')



while True:
	ret,frame = cap.read()

	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray_frame,1.3,5)
	if(len(faces)==0):
		continue

	for face in faces:
		x,y,w,h = face

		#Get the face ROI
		offset = 10
		face_section = gray_frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))

		#Predicted Label (out)
		out = knn(trainset,face_section.flatten())

		#Display on the screen the name and rectangle around it
		pred_name = name[int(out)]
		cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

	cv2.imshow("Faces",frame)

	key = cv2.waitKey(1) & 0xFF
	if key==ord('q'):
		break

cap.release()
cv2.destroyAllWindows()