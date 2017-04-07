
# coding: utf-8

# # Neural Nets Experiment
# 
# Basically what we do in a neural net:
# 1. We have a fixed input and a desired output
# 2. Create a random output, that will turn into predicted output, that we estimate by using a function
# 3. In every iteration we aproximate the predicted output to the desired output
# 4. When the output is close enough, we can extend the result to other inputs.

# In[1]:

import numpy as np

# examples of training data
training_input = np.array([[0,0,1], [1,1,1], [1,0,1], [0,1,0]])
#T because its the transposition of the array
training_output = np.array([[0,1,1,0]]).T


# In[2]:

class NeuralNetwork():
    def __init__(self):
        # random number generator
        np.random.seed(1)
        # single neuron with 3 input conections
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1
        # sigmoid function that receive the weighted sum of the inputs and normalize them between 0 and 1
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    # sigmoid derivative that indicates how confident is the random weights that we're created
    def __sigmoid_derivative(self, x):
        return x * (1 - x)
    def train(self, input_set, output_set, iterations):
        for iteration in iter(range(iterations)):
            # pass the input set through the neural net (that we fixed for one neuron)
            output = self.think(input_set)

            # calculate the error (diference between the desired and the predicted output)
            error = output_set - output

            # adjust the accuracy by multiplying the error by the input and by the derivate of the curve
            # that means that the most confident weights are adjusted, and those that don't need are not adjusted
            # np.dot(): matrix multiplication
            adjustment = np.dot(input_set.T, error * self.__sigmoid_derivative(output))

            # ajust the weigths for the next iteration
            self.synaptic_weights += adjustment

            if (iteration % 1000 == 0):
                print ("error after %s iterations: %s" % (iteration, str(np.mean(np.abs(error)))))
    # think function estimate the output
    def think(self, input_set):
        # pass the input to our neural net
        return self.__sigmoid(np.dot(input_set, self.synaptic_weights))


# In[3]:

if __name__ == '__main__':
    # initialize the neural net
    neural_net = NeuralNetwork()
    
    # generate the random output that will be estimated
    print ("Random starting synaptic weights: ")
    print (neural_net.synaptic_weights)
    
    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_net.train(training_input, training_output, 10000)
    
    # Test the neural network with a new pattern
    test = [0, 1, 0]
    print ("Considering new situation %s -> ?: " % test )
    print (neural_net.think(np.array(test)))

