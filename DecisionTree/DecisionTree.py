import graphviz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import tree
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split

cmap_bold = ListedColormap(['darkorange', 'darkblue'])

data = pd.read_csv('data.csv')
data = data[['diagnosis', 'compactness_mean', 'concavity_mean']]

data_new = data[-12:].to_numpy()
data = data[:-12]
data_training, data_testing, indices_train, indices_test = train_test_split(data, data.index,  test_size = 0.33, random_state = 0)
data_training = data_training.to_numpy()
data_testing = data_testing.to_numpy()

X_training = data_training[:, 1:3]
Y_training = data_training[:, 0]

X_testing = data_testing[:, 1:3]
Y_testing = data_testing[:, 0]

X_new = data_new[:, 1:3]
Y_new = data_new[:, 0]

plt.scatter(X_training[:, 0], X_training[:, 1], c=Y_training, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('compactness_mean')
plt.ylabel('concavity_mean')
plt.show()

clf = tree.DecisionTreeClassifier(max_depth=4).fit(X_training, Y_training)
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=['compactness_mean', 'concavity_mean'], class_names=['malignant', 'benign'], filled=True, rounded=True, special_characters=True) 
graph = graphviz.Source(dot_data)
graph.render('cancer')

disp = plot_confusion_matrix(clf, X_testing, Y_testing, display_labels = ['Benign', 'Malignant'], values_format='d')
plt.show()

plt.scatter(X_training[:, 0], X_training[:, 1], c=Y_training, cmap=cmap_bold, edgecolor='k', s=20)
plt.scatter(X_new[:, 0], X_new[:, 1], c='red', edgecolor='k', s=25)
plt.xlabel('compactness_mean')
plt.ylabel('concavity_mean')
plt.show()

Y_predicted = clf.predict(X_new)
Y_predicted[Y_predicted == 'B'] = 'Benign';
Y_predicted[Y_predicted == 'M'] = 'Malignant';
result = pd.DataFrame({'Compactness_mean':X_new[:, 0], 'Concavity_mean':X_new[:, 1], 'Predicted diagnosis':Y_predicted})
print(result)