
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from numpy import array,exp,dot,random 
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn import preprocessing

# In[2]:
def func(xt):

    data=pd.read_csv("/Users/vsvsvarma/Desktop/HL_pred/data.csv")


    y=data['Disease']
    x=data
    del x['Disease']


    # In[5]:


    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


    # In[6]:


    y_train=np.array(y_train)
    n=y_train.size
    y_train.resize(n,1)
    y_train



    # In[8]:


    class Neuronlayer():
        def __init__(self,input_nos,neuron_nos):
            self.weights=2*random.random((neuron_nos,input_nos))-1

    # In[9]:


    class MLP():
        
        def __init__(self,layer1,layer2,layer3):
            
            self.layer1=layer1
            self.layer2=layer2
            self.layer3=layer3
        
        
        def sigmoid(self,x,der):
            
            if(der==False):
                return 1 / (1 + exp(-x))
            else:
                return x*(1-x)
            
            
        def train(self,training_set_inputs,training_set_outputs,iterations):
            
            for iteration in range(iterations):
                
                
                output_layer1=self.sigmoid(dot(training_set_inputs,self.layer1.weights),False)
                output_layer2=self.sigmoid(dot(output_layer1,self.layer2.weights),False)
                output_layer3=self.sigmoid(dot(output_layer2,self.layer3.weights),False)
                
                
                
                #calculating the error 
                
                
                layer3_err=training_set_outputs-output_layer3
                layer3_delta=layer3_err*self.sigmoid(output_layer3,True)
                
                
                layer2_err=layer3_delta.dot(self.layer3.weights.T)
                layer2_delta=layer2_err*self.sigmoid(output_layer2,True)
                
                layer1_err=layer3_delta.dot(self.layer3.weights.T)
                layer1_delta=layer1_err*self.sigmoid(output_layer1,True)
                
                #calculating the adjustments
                
                alpha=0.06
                layer3_adjustments=alpha*(output_layer1.T.dot(layer3_delta))
                layer2_adjustments=alpha*(output_layer1.T.dot(layer2_delta))
                layer1_adjustments=alpha*(training_set_inputs.T.dot(layer1_delta))
                
                
                #adjusting the weights
                
                self.layer3.weights += layer3_adjustments
                self.layer2.weights += layer2_adjustments
                self.layer1.weights += layer1_adjustments
        
        def predict(self,inputs):
                output_layer1=self.sigmoid(dot(inputs,self.layer1.weights),False)
                output_layer2=self.sigmoid(dot(output_layer1,self.layer2.weights),False)
                output_layer3=self.sigmoid(dot(output_layer2,self.layer3.weights),False)
                return output_layer3
        
        def printw(self):
            
            print("Random Starting weights:")
            print(self.layer1.weights)
            print(self.layer2.weights)
            print(self.layer3.weights)
            


    # In[10]:

        
    layer1=Neuronlayer(10,10)
    layer2=Neuronlayer(10,10)
    layer3=Neuronlayer(1,10)
        
    neural_net=MLP(layer1,layer2,layer3)
        
        #neural_net.printw()
        
    neural_net.train(x_train,y_train,10000)

    xt=np.array(xt).reshape(1,-1)
    xts=preprocessing.normalize(xt)
    

    pred=neural_net.predict(xts)


    return pred;



