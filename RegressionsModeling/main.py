import numpy as np
import matplotlib.pyplot as plt
from ready import ReadyRegressionHandler
from custom import CustomRegressionHandler
from models.power_model import PowerModel
from models.linear_model import LinearModel
from models.parabolic_model import ParabolicModel
from models.hyperbolic_model import HyperbolicModel
from models.exponential_model import ExponentialModel

TEST_DATASET_SIZE = 100

data = np.genfromtxt('data.csv', delimiter=',')
weights = data[1:, 2:3]
heights = data[1:, 1:2]
r = np.corrcoef(weights.flatten(), heights.flatten())[0, 1]
print("Weights - Heights correlation coefficient: ", r)

max_r_2 = 0.
best_handler = None

rrh = ReadyRegressionHandler(TEST_DATASET_SIZE)
rrh.run(weights, heights)

linear_model = LinearModel()
crh = CustomRegressionHandler(linear_model, TEST_DATASET_SIZE)
crh.run(weights, heights)
max_r_2, best_handler = crh.compare(max_r_2, best_handler)

parabolic_model = ParabolicModel()
crh = CustomRegressionHandler(parabolic_model, TEST_DATASET_SIZE)
crh.run(weights, heights)
max_r_2, best_handler = crh.compare(max_r_2, best_handler)

exponential_model = ExponentialModel()
crh = CustomRegressionHandler(exponential_model, TEST_DATASET_SIZE)
crh.run(weights, heights)
max_r_2, best_handler = crh.compare(max_r_2, best_handler)

power_model = PowerModel()
crh = CustomRegressionHandler(power_model, TEST_DATASET_SIZE)
crh.run(weights, heights)
max_r_2, best_handler = crh.compare(max_r_2, best_handler)

hyperbolic_model = HyperbolicModel()
crh = CustomRegressionHandler(hyperbolic_model, TEST_DATASET_SIZE)
crh.run(weights, heights)
max_r_2, best_handler = crh.compare(max_r_2, best_handler)

print("The best model is", best_handler.model.__class__.__name__, "\nMean squared error is: ", best_handler.model.mse, "Predictions result:\n")
for (x, y, y_t) in best_handler.model.predictions: 
    #print(x, y, y_t)
    plt.scatter(x, y, c='#000000')
    plt.scatter(x, y_t, c='r')
plt.show()