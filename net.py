import numpy as np

#neuron function of activation and its derivative 
def sigmoid(x, deriv=False):
	if(deriv==True):
		return x*(1-x)
	return 1/(1+np.exp(-x))

#input data
x = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])

#expected output
y = np.array([[0],[1],[1],[0]])

#always generate same numbers
np.random.seed(1)

#synapses (connections between neurons)
#syn0 connects 3 input neurons to 4 neurons in middle layer
syn0 = 2*np.random.random((3,4)) - 1

#syn1 connects 4 neurons in middle layer to 1 output neuron
syn1 = 2*np.random.random((4,1)) - 1

#training
for i in range(60000):
	
	#inoput layer
	l0 = x
	#second layer is array of sigmoid functions of z, where each z is SUM(x*w) 
	#w is array of synapse weights
	#x is array of corresponding output from previous layer 
	l1 = sigmoid(np.dot(l0, syn0))
	l2 = sigmoid(np.dot(l1, syn1))

	#error is difference between expected output and output neuron l2
	l2_error = y - l2

	#print error rate to check it every 10000th iteration
	if(i%10000==0):
		print ("Error: " + str(np.mean(np.abs(l2_error))))

	#l2_delta used for calculating l1_error	
	l2_delta = l2_error*sigmoid(l2, deriv=True)

	#l1_error is array of error in second layer
	l1_error = l2_delta.dot(syn1.T)
	
	#l1_delta
	l1_delta = l1_error*sigmoid(l1, deriv=True)

	#update weights
	syn1 += l1.T.dot(l2_delta)
	syn0 += l0.T.dot(l1_delta)	

print("Output after training: ")
print(l2)

