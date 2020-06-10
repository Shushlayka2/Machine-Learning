import nltk
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, plot_confusion_matrix 
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

#data fetching
dataset = pd.read_csv('data\BBC News Train.csv')
categories = dataset[['Category']].drop_duplicates().sort_values('Category').Category.values

dataset['Category'] = dataset['Category'].factorize()[0]
dataset = dataset[:-4]
new_dataset = dataset[-4:]

#data preprocessing
punctuations = string.punctuation.translate({ord('\''): None})
stopwords = nltk.corpus.stopwords.words('english')
wn = nltk.WordNetLemmatizer()

def clean_txt(txt):
    words = "".join([c for c in txt if c not in punctuations]).lower().split()
    cleaned_words = [word for word in words if word not in stopwords]
    lemmatized_words = [wn.lemmatize(word) for word in cleaned_words]
    return lemmatized_words

#data vectorization
cv = CountVectorizer(analyzer = clean_txt)
x = cv.fit_transform(dataset['Text'])
y = dataset['Category']
x_train, x_test, y_train, y_test, indices_train, indices_test = train_test_split(x, y, dataset.index, test_size = 0.33, random_state = 0)

#data visualization
np.random.seed(0)
used_features_count = 7500
colors = ['blue', 'green', 'red', 'yellow', 'brown']
category_ids = {'business':0, 'tech':1, 'politics':2, 'sport':3, 'entertainment':4}
ids = np.random.choice(range(x.shape[0]), size = used_features_count)
tsne = TSNE(n_components=2, random_state=0)
visual_articles = tsne.fit_transform(x[ids])
for category, category_id in category_ids.items():
    points = visual_articles[y[ids] == category_id]
    plt.scatter(points[:, 0], points[:, 1], s=30, c=colors[category_id], label=category)
plt.legend()
plt.show()

#training
model = MultinomialNB()
model.fit(x_train, y_train)

#test
disp = plot_confusion_matrix(model, x_test, y_test, display_labels = categories, values_format='d')
plt.show()

#new articles
x_test = cv.transform(new_dataset['Text'])
y_test = new_dataset['Category']

y_test_pred = model.predict(x_test)
print(confusion_matrix(y_test, y_test_pred))

#new articles representation
for category, category_id in category_ids.items():
    points = visual_articles[y[ids] == category_id]
    plt.scatter(points[:, 0], points[:, 1], s=30, c=colors[category_id], label=category)
visual_articles = tsne.fit_transform(x_test)
plt.scatter(visual_articles[:, 0], visual_articles[:, 1], s=30, c='black')
plt.legend()
plt.show()