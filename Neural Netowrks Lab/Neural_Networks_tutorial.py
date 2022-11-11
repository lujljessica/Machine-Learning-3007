import numpy as np
import matplotlib.pyplot as plt
import NeuralNetwork as NN


def classify_point(x1, x2):
	if f(x1) > x2:
		return(0)
	else:
		return(1)
		
def classify_points(x1, x2):
	labels = []
	for i in range(x1.shape[0]):
		label = classify_point(x1[i], x2[i])
		labels.append(label)
	labels = np.asarray(labels)
	return(labels)


def f(x):
	ans = x**2 * np.sin(2 * np.pi * x) + 0.7
	return(ans)

def gauss_2d(mu_x, sigma_x, mu_y, sigma_y, size):
	x = np.random.normal(mu_x, sigma_x, size)
	y = np.random.normal(mu_y, sigma_y, size)
	return(x,y)

def plot_points(x1, x2, labels ):
	ones = np.where(labels[:] == 1)[0]
	zeros = np.where(labels[:] == 0)[0]
	plt.plot(x1[zeros],x2[zeros] , 'x')
	plt.plot(x1[ones],  x2[ones], 'o')
	plt.show()


def main():
	x1, x2 = gauss_2d(0.5, 0.15, 0.5, 0.15, 100)
	labels = classify_points(x1, x2)
#	plot_points(x1, x2, labels)
	# initialize or set up set up architecture of the neural network ie num layers and neurons
	network = NN.neural_network()
	network.init_network([2,3,4,1])  # input layer, hidden layer, output layer . Value is num neurons in each including bias
 
	# Train network
	epsilon = 0.05
	network.train(x1,x2, labels, 0.01, 1000, epsilon, 0.002, 100)
	# generate validation 
	v1, v2 = gauss_2d(0.5, 0.15, 0.5, 0.15, 100)
	labels_v = classify_points(v1, v2)
	corr_ones,  corr_zeros, inc_one, inc_zero = network.predict(v1,v2,labels_v)
	ones = np.where(labels_v[:] == 1)[0]
	zeros = np.where(labels_v[:] == 0)[0]

	print("Confusion Matrix"+"\n");
	print("__________________________");
 
	print("|        | Pred Zeros  | Pred Ones  |");
	print("________________________________________");
	print("| True Zeros  |     %d      |     %d      |" %(corr_zeros, inc_zero));
	print("| True Ones  |     %d      |     %d      |" %(inc_one, corr_ones));
	print("________________________________________");
	# generate validation 
	t1, t2 = gauss_2d(0.5, 0.15, 0.5, 0.15, 100)
	labels_t = classify_points(t1, t2)
	#network.predict(t1,t2,labels_t)
main()