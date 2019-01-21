import os
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.color import rgb2gray
from sklearn import svm
from sklearn.metrics import accuracy_score
from natsort import natsorted
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pylab as pl
f = open("ans.out", 'w')
sys.stdout = f

from numpy.random import seed
np.random.seed(20)

size=20

class Data:
	element=[[0 for x in range(size)] for y in range(size)]
	label=0
	def __init__(self,e,l):
		self.element=e
		self.label=l

train=[]
mainPath='TrainingData'
total=36

# A-10 , B-11....
i=0
for subDir in os.listdir(mainPath):
	path=os.path.join(mainPath,subDir)
	for fileName in os.listdir(path):
		filePath=os.path.join(path,fileName)
		image=mpimg.imread(filePath)
		image=rgb2gray(image)
		obj=Data(image,i)
		train.append(obj)
	i=i+1
l=len(train)

np.random.shuffle(train)

trainingSet,testSet=np.split(train,[int(0.80*l)])

print('Training_set=',len(trainingSet))
print('Test_set=',len(testSet))


xTrain=[]
yTrain=[]

for e in trainingSet:
	x=e.element
	y=e.label
	x=x.flatten()
	xTrain.append(x)
	yTrain.append(y)

xTest=[]
yTest=[]

for e in testSet:
	x=e.element
	y=e.label
	x=x.flatten()
	xTest.append(x)
	yTest.append(y)

clf = KNeighborsClassifier(n_neighbors=3)
#clf = svm.SVC(kernel='linear', C = 1.0,probability=True)

clf.fit(xTrain,yTrain)
limit=len(xTest)
y_pred=clf.predict(xTest);

#print(confusion_matrix(yTest,y_pred))  
#print(classification_report(yTest,y_pred)) 

print('Training accuracy=',accuracy_score(yTrain,clf.predict(xTrain)))
print('Test accuracy=',accuracy_score(yTest,clf.predict(xTest)))


temp=[]
listOfFiles=[]

mainPath='dataset'

dirFiles=os.listdir(mainPath)
dirFiles=natsorted(dirFiles)


check=0
for subDir in dirFiles:
	path=os.path.join(mainPath,subDir)
	image=mpimg.imread(path)
	image=rgb2gray(image)
	image=image.flatten()
	listOfFiles.append(subDir)
	temp.append(image)


prob=clf.predict_proba(temp)
#print(prob)
predictions=clf.predict(temp)

#print(predictions)
#print('Reconition:')
for i in range(len(prob)):
	maxElement=np.amax(prob[i])
	if(maxElement<.11):
		continue
	val=int(predictions[i])
	ans='' 
	if(val>=0 and val<=9):
		ans=str(val)
	else:
		diff=val-10
		ans=chr(ord('A')+diff)
	#print(maxElement)
	print(ans);
	#print(ans,listOfFiles[i])