import numpy as np
import matplotlib.pyplot as plt
from models.model import Model

class ExponentialModel(Model):
    
    def train(self):
        y_ln = np.log(self.y_train)
        x_y_ln = self.x_train * y_ln
        y_ln_avg = np.average(y_ln)
        x_y_ln_avg = np.average(x_y_ln)
        eq_left_side = np.array([[1, self.x_avg], [self.x_avg, self.x_2_avg]])
        eq_right_side = np.array([y_ln_avg, x_y_ln_avg])
        params = np.linalg.solve(eq_left_side, eq_right_side)
        self.a = params[0]
        self.b = params[1]

    def test(self):
        y_t = np.exp(self.a + self.b * self.x_test)
        e = y_t - self.y_test
        e_pos_avg = np.average(e[e > 0])
        e_neg_avg = np.average(e[e < 0])
        y_t_e_pos = y_t + e_pos_avg
        y_t_e_neg = y_t + e_neg_avg

        plt.plot(self.x_test, y_t, 'b')
        plt.plot(self.x_test, y_t_e_pos, 'r--')
        plt.plot(self.x_test, y_t_e_neg, 'r--')
        r = np.corrcoef(self.y_test, y_t)[0, 1]
        self.r_2 = r * r
        print("Exponential Model:\nCorrelation coefficient: ", r, "\nDetermination coefficient: ", self.r_2)

    def calculate_mse(self, x, y):
        y_t = np.exp(self.a + self.b * x)
        self.mse = sum((y - y_t)**2)
        self.predictions = zip(x, y, y_t)