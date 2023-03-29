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
    import keras
    import numpy as np
    from PIL import Image
    
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
    
    # Load the image and convert it to grayscale
    image = Image.open('test_image.png').convert('L')
    # Resize the image to 28x28 pixels
    image = image.resize((28, 28))
    # Flatten the image to a 1D array of length 784
    image_array = np.array(image).reshape(1, 784)
    # Normalize the pixel values to be between 0 and 1
    image_array = image_array / 255.0
    # Pass the flattened and normalized image through the trained neural network and get the predicted class
    predicted_class = np.argmax(model.predict(image_array))
    
    print("Predicted digit:", predicted_class)


def SVMMNIST():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    
    # Load the MNIST dataset
    digits = load_digits()

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

    # Train SVM with different kernels
    svc_linear = SVC(kernel='linear')
    svc_linear.fit(X_train, y_train)
    print("Accuracy of linear kernel:", svc_linear.score(X_test, y_test))

    svc_rbf = SVC(kernel='rbf')
    svc_rbf.fit(X_train, y_train)
    print("Accuracy of RBF kernel:", svc_rbf.score(X_test, y_test))

    # Train MLP neural network
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=10, alpha=1e-4, 
                        solver='sgd', verbose=10, random_state=1,
                        learning_rate_init=.1)

    mlp.fit(X_train, y_train)

    print("Accuracy of MLP neural network:", mlp.score(X_test, y_test))

def ID3():
    import pandas as pd
    import math
    import numpy as np

    # Load the PlayTennis dataset
    data = pd.read_csv("PlayTennis.csv")

    # Define the ID3 algorithm function
    def id3(data, target_attribute_name, attribute_names, default_class=None):
        # Count the number of each target class in the data
        classes, class_counts = np.unique(data[target_attribute_name], return_counts=True)

        # If all the data belongs to a single class, return that class
        if len(classes) == 1:
            return classes[0]

        # If there are no attributes left to split on, return the default class
        if len(attribute_names) == 0:
            return default_class

        # Calculate the entropy of the current data
        entropy = calculate_entropy(data[target_attribute_name], class_counts)

        # Initialize variables for tracking the best attribute and its information gain
        best_info_gain = -1
        best_attribute = None

        # Loop over all attributes and calculate their information gain
        for attribute in attribute_names:
            attribute_values, attribute_counts = np.unique(data[attribute], return_counts=True)

            # Calculate the weighted entropy of each possible value of the attribute
            weighted_entropy = 0
            for i in range(len(attribute_values)):
                subset = data[data[attribute] == attribute_values[i]]
                subset_classes, subset_class_counts = np.unique(subset[target_attribute_name], return_counts=True)
                weighted_entropy += (attribute_counts[i] / len(data)) * calculate_entropy(subset[target_attribute_name], subset_class_counts)

            # Calculate the information gain of the attribute
            info_gain = entropy - weighted_entropy

            # Update the best attribute and its information gain if this one is better
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_attribute = attribute

        # Create a new decision tree node with the best attribute
        tree = {best_attribute: {}}

        # Remove the best attribute from the list of attribute names
        attribute_names = [attr for attr in attribute_names if attr != best_attribute]

        # Loop over the possible values of the best attribute and create a subtree for each one
        for value in np.unique(data[best_attribute]):
            # Recursively call the ID3 algorithm on the subset of data with this value of the best attribute
            subtree = id3(data[data[best_attribute] == value], target_attribute_name, attribute_names, default_class)

            # Add this subtree to the main decision tree node
            tree[best_attribute][value] = subtree

        # If the default class is not None, add a subtree for missing attribute values
        if default_class is not None:
            tree["default"] = default_class
        return tree

    # Define a function to calculate the entropy of a set of target classes
    def calculate_entropy(target_attribute, class_counts):
        entropy = 0
        total = sum(class_counts)
        for count in class_counts:
            probability = count / total
            entropy -= probability * math.log2(probability)
        return entropy

    # Define the attribute names and target attribute for the PlayTennis dataset
    attribute_names = ["Outlook", "Temperature", "Humidity", "Windy"]
    target_attribute_name = "PlayTennis"

    # Run the ID3 algorithm on the PlayTennis dataset
    decision_tree = id3(data, target_attribute_name, attribute_names)

    # Print the resulting decision tree
    print(decision_tree)