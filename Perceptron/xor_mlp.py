import numpy as np

np.random.seed(42)

X = np.array([
    [0,0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([[0],[1],[1],[0]])

#first layer 
W1 = np.random.randn(2, 4)
b1 = np.random.randn(1, 4)

#second layer 
W2 = np.random.randn(4, 2)
b2 = np.random.randn(1, 2)

#output layer 
W3 = np.random.randn(2, 1)
b3 = np.random.randn(1, 1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


Z1 = X @ W1 + b1
A1 = sigmoid(Z1)

Z2 = A1 @ W2 + b2
A2 = sigmoid(Z2)

Z3 = A2 @ W3 + b3
A3 = sigmoid(Z3)


print("Input X:\n", X)
print("\nHidden Layer 1 output (4 neurons):\n", A1)
print("\nHidden Layer 2 output (2 neurons):\n", A2)
print("\nFinal output:\n", A3)
print("\nActual XOR labels:\n", y)

#it currently dosent produce the correct output for XOR, 
# but it demonstrates the structure of a multi-layer perceptron with two hidden layers. 
# To train this network to correctly model the XOR function, you would need to implement a training loop with backpropagation and an appropriate loss function.