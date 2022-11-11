from ast import Assign
from hmac import new
import numpy as np
import pandas as pd
import layers as layers

class neural_network:
	def __init__(self):
		self._layers = []
		return
	
	def init_network(self, layer_size_list):
		"""
		Initializes the network according to a list of layer sizes - ie num neurons
		"""
		# Iterate through list of layer sizes then create layer
		self.layer_size_list = layer_size_list
		self.num_layers = len(layer_size_list)
		for i in range(len(layer_size_list)):
			if(layer_size_list[i] != layer_size_list[-1]):
				layer = layers.Layer((layer_size_list[i]+1), layer_size_list[i + 1]) 
			else:
				layer = layers.Layer(layer_size_list[-1],1)
			self._layers.append(layer) # adds layer to the network
		return
	
	def train(self, x1, x2, y, learning_rate, max_epochs, epsilon, _lambda, n):
		""" Get all activations and results with forward propagation. 
			Take in x1 and x2 as inputs.
			learning_rate: alpha 
			epochs - num cycles
		"""
		# two terminating conditions in outer loop
		for i in range(max_epochs):
			j = 1
			for j in range(self.num_layers):
				layer = self._layers[j]
				num_errors = self.layer_size_list[j]
				layer.gradient = np.zeros( num_errors+1, dtype= float)
	
			for j in range(len(x1)):
				x = np.asarray([x1[j], x2[j]]) # making it an array to do the vectorized calculations 
				self.forward_prop(x)
				self.backward_prop(x,y[j])
				self.compute_reg_gradients(_lambda,n)
				norm_diff = self.update_weights(learning_rate) # update weights and get the norm difference of all new and old weights
				if(np.mean(norm_diff) < epsilon):
					return
		return
	
	def forward_prop(self, x):
		""" Forward Propagation to get all activations / outputs."""
		# Iterate through each layer and activate the layer using the previous result as input to the next iteration
		curr_act = x
		bias = True
		keep_res = False
		for layer in self._layers:
			if (layer == self._layers[-1]):
				keep_res = True			
				bias = False
			output = layer.activate(curr_act, bias)
			curr_act = output
			if (keep_res == True):
				return output
		return 
	
		
	def backward_prop(self, x, y): # y is the true labels.
		"""
		Performs the backpropagation algorithm and updates the layer weights accordingly.
			:param X: The input values.
			:param y: The target values.
			:param float learning_rate: The learning rate (between 0 and 1).
			get error in next layer. Mat mul with the activations in current layer. 
		"""
		# Loop over network in reverse
		for layer in reversed(self._layers): # l is the current layer in REVERSED order. i  is the oringal index of current layer
			if (layer == self._layers[-1]):
				# Last layer
				layer.error = (y - layer.activation)
				layer.gradient = np.asarray([0])
			else:
				i = self._layers.index(layer)
				next_layer = self._layers[i + 1]
				# curr layer error = mat Mul (transpose(weights)*next layer error) * sigmoid derivative on curr layer activations 
				activations = layer.activation
				temp = None
				temp_transpose =None
				
				if (next_layer == self._layers[-1]):
					t = np.transpose(layer.weights.flatten())
					temp = t*next_layer.error
					sigmoid = layer.apply_sigmoid_derivative(activations)
					layer.error = (temp*sigmoid)
					temp_transpose = np.transpose(activations)
					temp1 = np.multiply(next_layer.error,  temp_transpose)
					temp = np.add(layer.gradient,temp1)
					layer.gradient = np.asarray(temp)
				else:
					removed_bias_error = np.delete(next_layer.error,0)
					temp_transpose = np.transpose(layer.weights)
					temp = np.dot(temp_transpose, removed_bias_error )
					layer.error = np.multiply(temp,layer.apply_sigmoid_derivative(activations))
					temp_ts = np.transpose(activations)
					temp1 =  np.dot(removed_bias_error, temp_ts)
					temp = np.add(layer.gradient,temp1)
					layer.gradient = np.asarray(temp)
		return
	
	
	def compute_reg_gradients(self,_lambda,n):
	
		first = True
		for layer in self._layers:
			temp = []
			if (first == True):
				sum_axis=np.sum(layer.weights, axis =1)
			else:
				sum_axis= layer.weights.flatten()
			for r in range(len(layer.gradient)):
				if (r ==0 ):
					temp.append((1/n)*layer.gradient[r])
				else:
					sum = sum_axis[r]
					temp.append((1/n)*layer.gradient[r] + (_lambda/2*n)*(sum**2))
			first = False
			layer.reg_grad = np.asarray(temp)
		return

	def  update_weights(self, alpha):
		convergence = []
		for layer in self._layers:
			old_weights  = layer.weights
			if (np.mean(layer.error)< 0):
				new_weights = old_weights - alpha*(layer.reg_grad)
			elif (np.mean(layer.error)> 0):
				new_weights = old_weights + alpha*(layer.reg_grad)
			convergence.append(np.linalg.norm(new_weights - old_weights))
			
		return  np.asarray(convergence)

	def print_network_error(self):
		layer = self._layers[-1]
		print(layer.error)
		return

	def predict(self, x1,x2, true_labels):
		Correct_Pred_Ones = 0
		Correct_Pred_Zeros = 0
		Inc_Zero = 0 
		Inc_One =0
		for j in range(len(x1)):
			x = np.asarray([x1[j], x2[j]]) # making it an array to do the vectorized calculations 
			predicted_label_conf =self.forward_prop(x)
			predicted_label = self.assign_label(predicted_label_conf)

			if (predicted_label == true_labels[j] & true_labels[j] == 1):
				Correct_Pred_Ones += 1
			elif (predicted_label == true_labels[j] & true_labels[j] == 0):
				Correct_Pred_Zeros += 1		
			elif (predicted_label != true_labels[j] & true_labels[j] == 1):
				Inc_One +=  1
			elif (predicted_label != true_labels[j] & true_labels[j] == 0):
				Inc_Zero += 1
		return Correct_Pred_Ones, Correct_Pred_Zeros, Inc_One, Inc_Zero
	def assign_label(self, res):
		if (res > 0.5):
			return 1
		elif (res < 0.5): 
			return 0