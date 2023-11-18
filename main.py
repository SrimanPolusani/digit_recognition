# Import Statements
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import InputLayer, Dense
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
import matplotlib.pyplot as plt


class Artificial_Neural_Networks:
    def __init__(self):
        self.X_train = np.load("X.npy")
        self.y_train = np.load("y.npy")
        self.m, self.n = self.X_train.shape
        self.z = np.array([1., 2., 3., 4.])
        self.model = self.final_model()

    def final_model(self):
        """
        This method has the code for ANN model with three dense layers
        containing 25, 15 and 10 artificial neurons respectively
        :return: The ANN model
        """
        tf.random.set_seed(1234)
        model = Sequential(
            [
                InputLayer((400,)),
                Dense(25, activation="relu", name="L1"),  # Layer 1
                Dense(15, activation="relu", name="L2"),  # Layer 2
                Dense(10, activation="linear", name="L3")  # Layer 3
            ], name="my_model"
        )

        model.compile(
            loss=SparseCategoricalCrossentropy(from_logits=True),
            optimizer=Adam(learning_rate=0.001)
        )

        model.fit(self.X_train, self.y_train, epochs=40)
        return model

    def predictioner(self, index):
        """
        This method is used for making the predictions using the final model
        :param index: An nd array with a shape of (400,0)
        :return: An integer as a prediction for the given image
        """
        image_matrix = self.X_train[index]
        prediction = self.model.predict(image_matrix.reshape(1, 400))
        prediction_p = tf.nn.softmax(prediction)
        yhat = np.argmax(prediction_p)
        return yhat

    def softmax(self):
        """
        This method has code for finding softmax in three different ways
        N = Numbers of features in the input data
        :return: a, a_two, a_tf which are nd arrays with N values
        """
        # Softmax Approach 1
        ez = np.exp(self.z)
        a = ez / np.sum(ez)

        # Softmax Approach 2
        N = len(self.z)
        a_two = np.zeros(N)
        ez_sum = 0
        for k in range(N):
            ez_sum += np.exp(self.z[k])
        for j in range(N):
            a_two[j] = np.exp(self.z[j]) / ez_sum

        # Softmax Approach 3
        a_tf = tf.nn.softmax(self.z)

        return a, a_two, a_tf

    # Visualizing the Digits
    def visualizer(self, visualize_results: bool):
        """
        This method is used to both visualize the predictions of the final model
        or to just convert the matrices with pixel values in to images
        :param visualize_results: Boolean (True or False)
        :return: None
        """
        fig, axes = plt.subplots(8, 8, figsize=(5, 5))
        fig.tight_layout(pad=0.13, rect=[0, 0.03, 1, 0.91])

        for i, ax in enumerate(axes.flat):
            random_index = np.random.randint(0, self.m)
            X_reshaped = self.X_train[random_index].reshape(20, 20).T
            ax.imshow(X_reshaped, cmap='gray')
            if visualize_results:
                predicted_value = self.predictioner(random_index)
                ax.set_title("{}, {}".format(self.y_train[random_index, 0], predicted_value))
            else:
                ax.set_title("{}".format(self.y_train[random_index, 0]))
            ax.set_axis_off()
        if visualize_results:
            fig.suptitle("Actual, Prediction", fontsize=11)
        else:
            fig.suptitle("Label, Image", fontsize=11)
        plt.show()


# Instance of Object Artificial_Neural_Networks
ann = Artificial_Neural_Networks()

# Checking Softmax Values for All Approaches
softmax_values = ann.softmax()
for i, smax in enumerate(softmax_values):
    print("softmax with approach {}: {}".format(i + 1, smax))

# Visualizing Hand Written Digits
ann.visualizer(visualize_results=False)

# Visualizing the predictions of the model
ann.visualizer(visualize_results=True)
