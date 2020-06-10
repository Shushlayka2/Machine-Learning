import numpy as np

class Model:

    def set_train_data(self, x, y):
        self.x_train = x
        self.y_train = y
        self.x_avg = np.average(x)
        self.y_avg = np.average(y)
        self.x_2 = x * x
        self.x_y = x * y
        self.x_2_avg = np.average(self.x_2)
        self.x_y_avg = np.average(self.x_y)

    def set_test_data(self, x, y):
        self.x_test = x
        self.y_test = y

    def train(self):
        pass

    def test(self):
        pass

    def calculate_mse(self, x, y):
        pass