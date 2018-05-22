import numpy as np

# X = input
# Y = output
X = np.array(([0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 2], [1, 3], [2, 0], [2, 1], [2, 2], [2, 3], [3, 0], [3, 1],  [3, 2], [3, 3]), dtype=float)
y = np.array(([50], [20], [10], [0], [100], [100], [100], [100], [100], [100], [100], [100], [100], [100], [100], [100]), dtype=float)
xPredicted = np.array(([0, 1]), dtype=float)

y = y/100 # / 100 car pourcentage donc le max = 100

class Neural_Network(object):
  def __init__(self):
    #nombre de perceptron par couche
    self.inputSize = 2
    self.outputSize = 1
    self.hiddenSize = 2

    #poids entre couche 1 et 2 puis couche 2 et 3
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize)

  def forward(self, X):
    #forward propagation through our network
    self.z = np.dot(X, self.W1) # dot product of X (input) and first set of weights (inputSize * hiddenSize)
    self.z2 = self.sigmoid(self.z) # activation function
    self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of weights (hiddenSize * outputSize)
    o = self.sigmoid(self.z3) # final activation function
    return o

  def sigmoid(self, s):
    # activation function
    return 1/(1+np.exp(-s))

  def sigmoidPrime(self, s):
    #derivative of sigmoid
    return s * (1 - s)

  def backward(self, X, y, o):
    # backward propgate through the network
    self.o_error = y - o # error in output
    self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error

    self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
    self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error

    self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
    self.W2 += self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights

  def train(self, X, y):
    o = self.forward(X)
    self.backward(X, y, o)

  def saveWeights(self):
    np.savetxt("w1.txt", self.W1, fmt="%s")
    np.savetxt("w2.txt", self.W2, fmt="%s")

  def predict(self):
    print "Predicted data based on trained weights: ";
    print "Input (scaled): \n" + str(xPredicted);
    print "Output: \n" + str(self.forward(xPredicted)*100);

NN = Neural_Network()
for i in xrange(1000): # trains the NN 1,000 times
  # print "# " + str(i) + "\n"
  # print "Input (scaled): \n" + str(X)
  # print "Actual Output: \n" + str(y)
  # print "Predicted Output: \n" + str(NN.forward(X))
  # print "Loss: \n" + str(np.mean(np.square(y - NN.forward(X)))) # mean sum squared loss
  # print "\n"
  NN.train(X, y)

print "# " + str(i) + "\n"
print "Input (scaled): \n" + str(X)
print "Actual Output: \n" + str(y)
print "Predicted Output: \n" + str(NN.forward(X))
print "Loss: \n" + str(np.mean(np.square(y - NN.forward(X)))) # mean sum squared loss
print "\n"
NN.saveWeights()
NN.predict()