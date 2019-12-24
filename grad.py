import numpy as np
import matplotlib.pyplot as plt

def relu(x):
	return np.max(0.001,x)

def sigmoid(x):
	return np.divide(1,np.add(np.exp(-x),1))

def softmax(x):
	return np.exp(x)/np.sum(np.exp(x),axis=0)

gradient_func = {'sin':np.sin,'sinh':np.sinh,'cos':np.cos,'cosh':np.cosh,'tan':np.tanh,'tanh':np.tanh,'sigmoid':sigmoid,'relu':relu}

print('Support following functions:-',gradient_func.keys())



eps=1e-7

def gradient(func,x):
	return((gradient_func[func](x+eps))-(gradient_func[func](x-eps)))/(2*eps)