import numpy as np
import tensorflow as tf
from tensorflow import keras

def linearRegression():
    # Define your data points
    x_values = [1, 2, 3, 4, 5]
    y_values = [2, 4, 5, 4, 5]

    # Calculate the mean of x and y
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)

    # Calculate the slope and y-intercept of the line of best fit
    numerator = sum([(x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values)])
    denominator = sum([(x - x_mean) ** 2 for x in x_values])
    slope = numerator / denominator
    y_intercept = y_mean - slope * x_mean

    # Print the slope and y-intercept of the line of best fit
    print("Slope:", slope)
    print("Y-Intercept:", y_intercept)
    print("The final equation is: y = %5.2fx + %5.2f" % (slope, y_intercept))


def quadraticRegression():
    # Define your data points
    x_values = [1, 2, 3, 4, 5]
    y_values = [2, 4, 5, 4, 5]

    # Define the functions to calculate the coefficients a, b, and c of the quadratic equation y = ax^2 + bx + c
    def get_a(x_values, y_values):
        numerator = sum([y * x ** 2 for x, y in zip(x_values, y_values)]) * len(x_values)
        denominator = sum([x ** 4 for x in x_values]) * len(x_values) - (sum([x ** 2 for x in x_values])) ** 2
        return numerator / denominator

    def get_b(x_values, y_values, a):
        numerator = sum([y * x for x, y in zip(x_values, y_values)]) - a * sum([x ** 3 for x in x_values])
        denominator = sum([x ** 2 for x in x_values])
        return numerator / denominator

    def get_c(x_values, y_values, a, b):
        return sum(y_values) / len(x_values) - a * sum([x ** 2 for x in x_values]) - b * sum(x_values) / len(x_values)

    # Calculate the coefficients a, b, and c of the quadratic equation y = ax^2 + bx + c
    a = get_a(x_values, y_values)
    b = get_b(x_values, y_values, a)
    c = get_c(x_values, y_values, a, b)

    # Print the coefficients a, b, and c of the quadratic equation y = ax^2 + bx + c
    print("a:", a)
    print("b:", b)
    print("c:", c)
    print("The final equation is: y = %5.2fx^2 + %5.2fx +  %5.2f" % (a, b, c))

def XORClassification():
    # Define the sigmoid function and its derivative
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(x):
        return x * (1 - x)

    # Define the neural network class
    class NeuralNetwork:
        def __init__(self, input_size, hidden_size, output_size):
            self.weights1 = np.random.randn(input_size, hidden_size)
            self.weights2 = np.random.randn(hidden_size, output_size)

        def forward(self, X):
            self.hidden_layer = sigmoid(np.dot(X, self.weights1))
            self.output = sigmoid(np.dot(self.hidden_layer, self.weights2))
            return self.output

        def backward(self, X, y, learning_rate):
            output_error = y - self.output
            output_delta = output_error * sigmoid_derivative(self.output)

            hidden_error = output_delta.dot(self.weights2.T)
            hidden_delta = hidden_error * sigmoid_derivative(self.hidden_layer)

            self.weights2 += self.hidden_layer.T.dot(output_delta) * learning_rate
            self.weights1 += X.T.dot(hidden_delta) * learning_rate

        def train(self, X, y, learning_rate, epochs):
            for i in range(epochs):
                self.forward(X)
                self.backward(X, y, learning_rate)

    # Define the XOR input and output data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Initialize the neural network with 2 input neurons, 4 hidden neurons, and 1 output neuron
    nn = NeuralNetwork(2, 4, 1)

    # Train the neural network on the XOR data
    nn.train(X, y, 0.1, 100000)

    # Test the neural network on the XOR data
    for i in range(len(X)):
        print(X[i], y[i], nn.forward(X[i]))

def MNIST():

    # Load the MNIST dataset
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # Reshape and normalize the input data
    X_train = X_train.reshape(-1, 28 * 28) / 255.0
    X_test = X_test.reshape(-1, 28 * 28) / 255.0

    # Convert the output data to one-hot encoding
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    # Define the neural network model
    model = keras.models.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(784,)),
        keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model with categorical cross-entropy loss and Adam optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model on the MNIST data
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    # Evaluate the model on the test data
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print("Test loss:", test_loss)
    print("Test accuracy:", test_accuracy)

a = '-' * 100

# linearRegression()

# print('\n\n' + a + '\n\n')

# quadraticRegression()

# print('\n\n' + a + '\n\n')

XORClassification()

# print('\n\n' + a + '\n\n')

# MNIST()