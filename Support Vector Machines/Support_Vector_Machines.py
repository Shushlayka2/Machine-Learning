import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import plot_confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn import svm

h = .02
cmap_light = ListedColormap(['orange','cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'darkblue'])

data = pd.read_csv('data.csv')
data = data[['diagnosis', 'compactness_mean', 'concavity_mean']]
data['diagnosis'] = (data['diagnosis'] == 'B') * 1

data_training, data_testing, indices_train, indices_test = train_test_split(data, data.index,  test_size = 0.33, random_state = 0)
data = data.to_numpy()
data_training = data_training.to_numpy()
data_testing = data_testing.to_numpy()

X_training = data_training[:, 1:3]
Y_training = data_training[:, 0]

X_testing = data_testing[:, 1:3]
Y_testing = data_testing[:, 0]

plt.scatter(X_training[:, 0], X_training[:, 1], c=Y_training, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('compactness_mean')
plt.ylabel('concavity_mean')
plt.show()

clf = svm.SVC()
clf.fit(X_training, Y_training)

x_min, x_max = data[:, 1].min() - 0.1, data[:, 1].max() + 0.1
y_min, y_max = data[:, 2].min() - 0.1, data[:, 2].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z_training = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z_training = Z_training.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z_training, cmap=cmap_light)
plt.scatter(X_training[:, 0], X_training[:, 1], c=Y_training, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('compactness_mean')
plt.ylabel('concavity_mean')
plt.show()

plt.pcolormesh(xx, yy, Z_training, cmap=cmap_light)
plt.scatter(X_training[:, 0], X_training[:, 1], c=Y_training, cmap=cmap_bold, edgecolor='k', s=20)
plt.scatter(X_testing[:, 0], X_testing[:, 1], c='red', edgecolor='k', s=25)
plt.xlabel('compactness_mean')
plt.ylabel('concavity_mean')
plt.show()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
Y_predicted = clf.predict(X_testing)
Y_predicted = np.array(['Malignant', 'Benign'])[Y_predicted.astype('int')]
Y_testing_full = np.array(['Malignant', 'Benign'])[Y_testing.astype('int')]
result = pd.DataFrame({'Compactness_mean':X_testing[:, 0], 'Concavity_mean':X_testing[:, 1], 'Predicted diagnosis':Y_predicted, 'Real diagnosis':Y_testing_full, 'Is correct':Y_predicted==Y_testing_full})
print(result)

disp = plot_confusion_matrix(clf, X_testing, Y_testing, display_labels = ['Malignant', 'Benign'], values_format='d')
plt.show()