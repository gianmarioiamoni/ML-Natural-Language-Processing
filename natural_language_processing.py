# NATURAL LANGUAGE PROCESSING
#
# Sentiment analysis of resturant reviews
# 
# Training set is a set of 1000 reviews with a column with the
# indication of positive/negative (1/0)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
# Data file is in tsv (tab separeted values)
# The dataset requires some preprocessing, so we will create 
# training set and test set later
#
# delimiter: '\t' for tab delimiter
# quoting: we need to ignore the double quotes; always recommended in text processing
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
# We need to remove punctuations, stop words, etc.
#
# Simplify the text as much as possible
# We clean punctuation, special characters, etc.
#
# we use re library and nltk library: this import the stop words,
# the words not relevant for predicting the reviews ("the", "a", etc)
#
# We use also PorterStemming in order to apply stemming on reviews:
# It consists on of taking only the roor of a word that indicates enough
# about what this word means.
# i.e.: "Oh, I loved that resturant": "loved" => "love"
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 

N_REVIEWS = 1000

# corpus list wil contain all different reviews, all cleaned
corpus = []
# loop to iterate through all different reviews of the dataset

# Create a function which cleans a review and return the clened review
def clean_review(review):
  # Replace any letter that is not a letter by a space
  # ^ = not
  review = re.sub('[^a-zA-Z]', ' ', review)
  # Transform all capitol letter in lowercase
  review = review.lower()
  # Split the review into its differetn words
  review = review.split()
  # Apply stemming to each word of the review
  ps = PorterStemmer()
  # we skip stopwords; remove "not" fron the stop words
  all_stopwords = stopwords.words('english')    
  all_stopwords.remove('not')   
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  # join back the words together
  review = ' '.join(review)

  return review

for i in range(0, N_REVIEWS):
  # clean the i review

  review = clean_review(dataset['Review'][i])
  # add the cleaned review to the corpus
  corpus.append(review)

  # END FOR


# Creating the Bag of Words model
#
# We create the sparse matrix, which contains in the rows the different reviews
# we have in the corpus and in the columns all the different words taken 
# from all the different reviews.
# Each cell will get 0/1 depending on if the word of the column is-not/is 
# in the review on the row.
#
# This process in called Tokenization.
#
# We use the feature_extraction module from scikit-learn, in particular
# the CountVectorizer function
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) # max number of most frequent words = 1500
# put all the words of corpus and fit in the columns
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

len(X[0]) # 1565 words

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix and Accuracy Score
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

# Create a function which predicts the result of a new review
def predict_review(new_review):
  new_review = clean_review(new_review)

  new_corpus = [new_review]
  new_X_test = cv.transform(new_corpus).toarray()
  new_y_pred = classifier.predict(new_X_test)
  return new_y_pred

# Predict some new reviews
print(predict_review('I love this restaurant so much'))
print(predict_review('I hate this restaurant so much'))

print(predict_review('I prefer another place in the same location which is less expensive'))
print(predict_review('Service is bad but food is quite good'))



