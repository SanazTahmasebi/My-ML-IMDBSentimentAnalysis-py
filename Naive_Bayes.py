import os
import numpy as np
import nltk
from math import log
from operator import itemgetter
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import time

class Data(object):    
    def __init__(self, text=None, val=None):
        self.text = text
        self.val = val      

def MyNaive_Bayes_fit(X,Y):
    w0 = np.zeros([len(X[0])-1,1])
    w1 = np.zeros([len(X[0])-1,1])
    Ys = np.squeeze(Y)
    y0 = np.sum(Y)
    y1 = len(Y) - y0
    for j in range(len(X[0])-1):
        xj = X[:,j+1]
        x01 = (np.sum(np.logical_and(np.logical_not(xj),Ys))+1)/(y1+2)
        x00 = (np.sum(np.logical_and(np.logical_not(xj),np.logical_not(Ys)))+1)/(y0+2)
        w0[j]= log(x01/x00)
        x11 = (np.sum(np.logical_and(xj,Ys))+1)/(y1+2)
        x10 = (np.sum(np.logical_and(xj,np.logical_not(Ys)))+1)/(y0+2)
        w1[j]= log(x11/x10)
        
    return w0, w1
  
def MyNaive_Bayes_predict(w0,w1,Xv,Y):    
    Ypv = np.zeros([len(Xv),1])
    xvi = np.zeros([len(Xv[0])-1,1])
    y0 = np.sum(Y)
    y1 = len(Y) - y0
    for i in range(len(Xv)):
        xvi = np.array(np.transpose(Xv[i,1:])).reshape(len(Xv[0])-1,1)
        d = log(y1/y0)+np.sum(w0)+np.dot(np.squeeze(np.subtract(w1,w0)),np.squeeze(xvi))        
        if d>=0:
            Ypv[i]=1
        else:
            Ypv[i]=0
    return Ypv
      
        
pathN = 'E:/Courses/Applied Machine Learning/mini project 2/train/neg'
pathP = 'E:/Courses/Applied Machine Learning/mini project 2/train/pos'

Nt = len(os.listdir(pathN))+len(os.listdir(pathP))
N=int(0.8*Nt)
Nv=int(0.2*Nt)
Nw = 2500
Nk = 10
OffS = 0                # Offset to bypasss some frequent words
p = True
ExFea = 0               # Number of extra features: 3 or 5

MySW = ['br',"i'm",'2','1','10',"i'll",'b','5','\x96','4',"i'd",'oh',"he'","he's","there'"\
        "there's",'mr',"i've","i'v","that'","that's",'3','us','6','30',"we're"\
        "'the",'co','fi','e','20','9','de','7','8']

DataL = []
i = 0

"""
Data Pre-processing
"""

for filename in os.listdir(pathN):
    f = open("E:/Courses/Applied Machine Learning/mini project 2/train/neg/"+filename, "r",encoding="utf8")    
    a = f.read()
    DataL.append(Data(text=a,val=0))
    i = i+1    
    
for filename in os.listdir(pathP):
    f = open("E:/Courses/Applied Machine Learning/mini project 2/train/pos/"+filename, "r",encoding="utf8")
    a = f.read()
    DataL.append(Data(text=a,val=1))
    i = i+1

# Data Shuffling
for i in range(int(0.5*Nt)):
    if i%2!=0:
        continue;
    k = int(0.5*Nt) + i
    b = DataL[i]
    DataL[i] = DataL[k]
    DataL[k] = b

ss=''
for i in range(N):
    ss += DataL[i].text   

stop_words = set(stopwords.words('english')) 
porter = PorterStemmer()
lemmatizer = WordNetLemmatizer()

if p==True:
    table = str.maketrans(",.!?()-\"*<>/&;':",16*" ")    
    ss=ss.translate(table)    

Wc = dict()
ss = ss.lower()
ss2 = ss.split()
ss2 = [w for w in ss2 if not w in stop_words] 
ss3 = [w for w in ss2 if not w in MySW] 
#ss3 = [porter.stem(word) for word in ss3]
ss3 = [lemmatizer.lemmatize(word) for word in ss3]

for word in ss3:
    if word in Wc:
        Wc[word]+=1
    else:
        Wc[word]=1
        
WWc = sorted(Wc.items(), key=itemgetter(1), reverse = True)

topWc = WWc[OffS:OffS+Nw]    

Xt = np.zeros((N+Nv,Nw+ExFea+1))
Yt = np.zeros((N+Nv,1))

X = np.zeros((N,Nw+ExFea+1))
Y = np.zeros((N,1))

Xv = np.zeros((Nv,Nw+ExFea+1))
Yv = np.zeros((Nv,1))

for i in range(N+Nv):
    data_point = DataL[i]    
#    if i< N:        
    Yt[i] = data_point.val
    Xt[i][0]=1
    ss = data_point.text.lower()
    if p==True:
        table = str.maketrans(",.!?()-\"*<>/&;':",16*" ")    
        ss=ss.translate(table)   
    ss2 = ss.split()
    ss2 = [w for w in ss2 if not w in stop_words]         
    ss3 = [w for w in ss2 if not w in MySW] 
#    ss3 = [porter.stem(word) for word in ss3]
    ss3 = [lemmatizer.lemmatize(word) for word in ss3]
    for word in ss3:
        if any(word in code for code in topWc):
            j=[y[0] for y in topWc].index(word)   
            Xt[i][j+1]=1                      

"""
Naive Bayes
"""
   
X = Xt[0:N-1,:]
Y = Yt[0:N-1,:]
Xv = Xt[N:,:]
Yv = Yt[N:,:]
 
Ytr = np.ravel(Y)   
start1 = time.time() 
w0,w1=MyNaive_Bayes_fit(X,Ytr)
end1 = time.time()
start2 = time.time()
Ypv = MyNaive_Bayes_predict(w0,w1,Xv,Ytr)
end2 = time.time()
myAccuracy = np.sum(np.logical_not(np.logical_xor(Yv,Ypv)))/Nv
print(myAccuracy)
print("training time : " + str(end1 - start1))
print("validation time : " + str(end2 - start2))
