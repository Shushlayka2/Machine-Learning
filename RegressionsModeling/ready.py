import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

class ReadyRegressionHandler:

    def __init__(self, test_dataset_size):
        self.__regr = linear_model.LinearRegression()
        self.test_dataset_size = test_dataset_size

    def  run(self, x, y):
        x_train = x[:-self.test_dataset_size]
        x_test = x[-self.test_dataset_size:]

        y_train = y[:-self.test_dataset_size]
        y_test = y[-self.test_dataset_size:]

        self.__regr.fit(x_train, y_train)

        y_pred = self.__regr.predict(x_test)

        print('Coefficients: \n', self.__regr.coef_)

        print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
        print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))

        # Plot outputs
        plt.scatter(x_test, y_test,  color='black')
        plt.plot(x_test, y_pred, color='blue', linewidth=3)

        plt.xticks(())
        plt.yticks(())

        plt.show()