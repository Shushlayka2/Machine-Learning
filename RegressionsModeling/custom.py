import random
import numpy as np
import matplotlib.pyplot as plt
from models.model import Model

class CustomRegressionHandler:
    
    def __init__(self, model, test_dataset_size):
        self.model = model
        self.test_dataset_size = test_dataset_size

    def __normalize(self, data):
        avg = np.average(data)
        inf = np.min(data) - 0.05 * avg
        sup = np.max(data) + 0.05 * avg
        dist = sup - inf
        data = (data - inf) / dist
        return data
        
    def __draw_confidence_bar(self, x, y):
        x_avg = np.average(x)
        y_avg = np.average(y)
        x_l = x - x_avg
        y_l = y - y_avg

        x_l_neg_avg = np.average(x_l[x_l < 0])
        y_l_neg_avg = np.average(y_l[y_l < 0])
        x_l_pos_avg = np.average(x_l[x_l > 0])
        y_l_pos_avg = np.average(y_l[y_l > 0])
        self.x_h_neg = x_avg + x_l_neg_avg
        self.y_h_neg = y_avg + y_l_neg_avg
        self.x_h_pos = x_avg + x_l_pos_avg
        self.y_h_pos = y_avg + y_l_pos_avg

        left_side = min(x[-self.test_dataset_size:])
        right_side = max(x[-self.test_dataset_size:])
        bottom_side = min(y[-self.test_dataset_size:])
        top_side = max(y[-self.test_dataset_size:])
        
        plt.plot(np.array([left_side, right_side]), np.repeat(self.x_h_neg, 2), 'r--')
        plt.plot(np.array([left_side, right_side]), np.repeat(self.x_h_pos, 2), 'r--')
        plt.plot(np.repeat(self.y_h_neg, 2), np.array([bottom_side, top_side]), 'r--')
        plt.plot(np.repeat(self.y_h_pos, 2), np.array([bottom_side, top_side]), 'r--')    

    def run(self, x, y):
        x = x.flatten()
        y = y.flatten()
        x_norm = self.__normalize(x)
        y_norm = self.__normalize(y)

        x_train = x_norm[:-self.test_dataset_size]
        x_test = x_norm[-self.test_dataset_size:]
        y_train = y_norm[:-self.test_dataset_size]
        y_test = y_norm[-self.test_dataset_size:]

        indexes = x_test.argsort()
        x_test = x_test[indexes]
        y_test = y_test[indexes]

        # other way of parallel sorting
        # x_test, y_test = zip(*sorted(zip(x_test, y_test)))
        # x_test = np.array(x_test)
        # y_test = np.array(y_test)

        plt.scatter(x_test, y_test, c='#000000')
        self.__draw_confidence_bar(x_train, y_train)
        self.model.set_train_data(x_train, y_train)
        self.model.set_test_data(x_test, y_test)

        self.model.train()
        result = self.model.test()
        plt.show()
        x_norm = self.__normalize(x)
        self.calculate_mse(x_norm, y_norm, 9000)
        return result

    def calculate_mse(self, x, y, count):
        x, y = zip(*random.choices(list(filter(lambda point: (point[0] > self.x_h_neg and point[0] < self.x_h_pos and point[1] > self.y_h_neg and point[1] < self.y_h_pos), zip(x, y))), k = count))
        x = np.array(x)
        y = np.array(y)
        self.model.calculate_mse(x, y)

    def compare(self, max_r_2, best_handler):
        if max_r_2 < self.model.r_2:
            max_r_2 = self.model.r_2
            best_handler = self
        return max_r_2, best_handler