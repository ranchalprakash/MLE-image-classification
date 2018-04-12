import pandas as pd
import numpy as np
print('loading train')
df = pd.read_csv('P1_data_train.csv',header=None)
df[len(df.columns)] =pd.read_csv('P1_labels_train.csv',header=None)

five=df.loc[df.iloc[:,-1] == 5]
u5=np.mean(five.iloc[:,0:-1])
cov5=np.cov(five.iloc[:,0:-1].T)

six=df.loc[df.iloc[:,-1]==6]
u6=np.mean(six.iloc[:,0:-1])
cov6=np.cov(six.iloc[:,0:-1].T)


tf = pd.read_csv('P1_data_test.csv',header=None)
pred= np.zeros(len(tf))

cov5_inv = np.linalg.inv(cov5)
cov6_inv = np.linalg.inv(cov6)

cov5_det = np.linalg.det(cov5)
cov6_det = np.linalg.det(cov6)

#covi=np.cov(df.iloc[:,0:-1].T)
covi=(cov5+cov6)/2
covi_inv = np.linalg.inv(covi)
covi_det = np.linalg.det(covi)

for k in range(0,len(tf)):
    g5 =  - 0.5*np.log(covi_det) + np.log(len(five)/len(df)) - 0.5*np.dot(np.dot(( tf.iloc[k,:] - u5), covi_inv), ( tf.iloc[k,:]- u5).T)
    g6 =  - 0.5*np.log(covi_det) + np.log(len(six)/len(df)) - 0.5*np.dot(np.dot(( tf.iloc[k,:] - u6), covi_inv), (tf.iloc[k,:] - u6).T)    
    
#    g5 =  - 0.5*np.log(cov5_det) + np.log(len(five)/len(df)) - 0.5*np.dot(np.dot(( tf.iloc[k,:] - u5), cov5_inv), ( tf.iloc[k,:]- u5).T)
#    g6 =  - 0.5*np.log(cov6_det) + np.log(len(six)/len(df)) - 0.5*np.dot(np.dot(( tf.iloc[k,:] - u6), cov6_inv), (tf.iloc[k,:] - u6).T)    
#  
    if g5<g6:
        pred[k]=6
    else:
        pred[k]=5
        
crct = 0
false6 = 0
false5 = 0
true5 = 0
true6= 0
label=pd.read_csv('P1_labels_test.csv',header=None)
for p in range(len(label)):
    if label.iloc[p,0]-pred[p]==0:
        if label.iloc[p,0] == 5:
            true5 += 1
        else:
            true6 += 1
        crct += 1
    elif label.iloc[p,0]-pred[p] == -1:
        false6 += 1
    else:
        false5+= 1
confusion_matrix = np.array([[true5, false6],[false5, true6]])     
        

        