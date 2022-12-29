import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

true = pd.read_csv('/content/True politics.csv')
fake = pd.read_csv('/content/Fake politics.csv')

print(true)
print(fake)

fake['target'] = 'fake'
true['target'] = 'true'

data = pd.concat([fake, true]).reset_index(drop = True)

from sklearn.utils import shuffle
data = shuffle(data)
data = data.reset_index(drop=True)

data.drop(["date"],axis=1,inplace=True)
data.drop(["title"],axis=1,inplace=True)
data['text'] = data['text'].apply(lambda x: x.lower())

import string

def punctuation_removal(text):
    all_list = [char for char in text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str

data['text'] = data['text'].apply(punctuation_removal)

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')

data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

from wordcloud import WordCloud

fake_data = data[data["target"] == "fake"]
all_words = ' '.join([text for text in fake_data.text])

wordcloud = WordCloud(width= 800, height= 500,
                          max_font_size = 110,
                          collocations = False).generate(all_words)

plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

real_data = data[data["target"] == "true"]
all_words = ' '.join([text for text in fake_data.text])

wordcloud = WordCloud(width= 800, height= 500,
                          max_font_size = 110,
                          collocations = False).generate(all_words)

plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

from nltk import tokenize
token_space = tokenize.WhitespaceTokenizer()
from sklearn import metrics
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

X_train,X_test,y_train,y_test = train_test_split(data['text'], data.target, test_size=0.2, random_state=42)
from sklearn.tree import DecisionTreeClassifier
pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', DecisionTreeClassifier(criterion= 'entropy',
                                           max_depth = 20, 
                                           splitter='best', 
                                           random_state=42))])

model = pipe.fit(X_train, y_train)
prediction = model.predict(X_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
cm = metrics.confusion_matrix(y_test, prediction)
plot_confusion_matrix(cm, classes=['Fake', 'Real'])

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from functools import partial


class NLTKVectorizer(TfidfVectorizer):

    def __init__(self, input='content', encoding='utf-8', decode_error='strict', strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, analyzer='word', stop_words=None, token_pattern='(?u)\\b\\w\\w+\\b', ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False, dtype=np.float64, norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False, min_token_length=3):
        super().__init__(input=input, encoding=encoding, decode_error=decode_error, strip_accents=strip_accents, lowercase=lowercase, preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer, stop_words=stop_words,
                         token_pattern=token_pattern, ngram_range=ngram_range, max_df=max_df, min_df=min_df, max_features=max_features, vocabulary=vocabulary, binary=binary, dtype=dtype, norm=norm, use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = stop_words
        self.min_token_length = min_token_length

    def tokenize_document(self, document):
        '''
        tokenize a document
        '''

        document_tokens = word_tokenize(document)

        return document_tokens

    def remove_non_alphanumeric_tokens(self, tokens_list):
        '''
        remove the tokens that are not alphanumeric (e.g. punctuations, special characters, etc ...)
        '''

        alphanumeric_tokens = [
            token for token in tokens_list if token.isalnum()]

        return alphanumeric_tokens

    def remove_numeric_tokens(self, tokens_list):
        '''
        remove the tokens that consists of only numeric characters
        '''

        non_numeric_tokens = [
            token for token in tokens_list if not token.isnumeric()]

        return non_numeric_tokens

    def lemmatize_document(self, tokens_list):
        '''
        use `NLTK` `WordNetLemmatizer` to lemmatize tokens
        '''

        lemmas = [self.lemmatizer.lemmatize(token) for token in tokens_list]

        return lemmas

    def convert_to_lowercase(self, document_tokens):
        '''
        convert tokens to lowercase
        '''

        lowercase_tokens = [token.lower() for token in document_tokens]

        return lowercase_tokens

    def remove_stop_words(self, document_tokens):
        '''
        remove stopwords from the document tokens
        '''
        
        if self.stop_words is not None:
            tokens = [token for token in document_tokens if token not in self.stop_words]
            return tokens
        else:
            return document_tokens

    def remove_short_tokens(self, document_tokens):
        '''
        keep only tokens whose length is higher than predefined threshold
        '''

        tokens = [token for token in document_tokens if len(
            token) > self.min_token_length]

        return tokens

    def analyze_document(self, document):
        '''
        preform some pre-processing steps on the document
        and extract the most *important* words from the document
        '''

        # tokenize the document
        document_tokens = self.tokenize_document(document)

        # remove tokens which are not alpha numeric
        document_tokens = self.remove_non_alphanumeric_tokens(document_tokens)

        # remove only numeric tokens
        document_tokens = self.remove_numeric_tokens(document_tokens)

        # lemmatize tokens
        document_tokens = self.lemmatize_document(document_tokens)

        # convert tokens to lower case
        document_tokens = self.convert_to_lowercase(document_tokens)

        # remove stopwords
        document_tokens = self.remove_stop_words(document_tokens)

        # filter *short* tokens
        document_tokens = self.remove_short_tokens(document_tokens)

        return document_tokens

    def build_analyzer(self):
        return lambda document: self.analyze_document(document)

from sklearn.svm import LinearSVC

vectorizer = NLTKVectorizer(
    max_df=0.5,
    min_df=10,
    ngram_range=(1, 1),
    max_features=10000,
    stop_words=stop,
)

svm_clf = LinearSVC(C=1.0)

from sklearn.calibration import CalibratedClassifierCV
clf = CalibratedClassifierCV(base_estimator=svm_clf, cv=5, method="isotonic")
pipeline = Pipeline([("vect", vectorizer), ("clf", clf)])

import nltk
nltk.download('punkt')

import nltk
nltk.download('wordnet')

import nltk
nltk.download('omw-1.4')

from sklearn.linear_model import LogisticRegression

vectorizer = NLTKVectorizer(stop_words=stop,
                            max_df=0.5, min_df=10, max_features=10000)

# fit on all the documents
vectorizer.fit(data['text'])

# vectorize the training and testing data
X_train_vect = vectorizer.transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# logistic Regression classifier
lr_clf = LogisticRegression(C=1.0, solver='newton-cg', multi_class='multinomial')

# fit on the training data
lr_clf.fit(X_train_vect, y_train)

# predict using the test data
y_pred = lr_clf.predict(X_test_vect)

from sklearn.decomposition import TruncatedSVD

vectorizer = NLTKVectorizer(stop_words=stop,
                            max_df=0.5, min_df=10, max_features=10000)

# fit on all the documents
vectorizer.fit(data['text'])

# vectorize the training and testing data
X_train_vect = vectorizer.transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# dimensionality reduction transformer, reduce the vector dimension to only 100
svd = TruncatedSVD(n_components=100)

# reduce the features vector space
X_train_vect_reduced = svd.fit_transform(X_train_vect)
X_test_vect_reduced = svd.fit_transform(X_test_vect)

# logistic Regression classifier
lr_clf = LogisticRegression(C=1.0, solver='newton-cg', multi_class='multinomial')

# fit on the training data
lr_clf.fit(X_train_vect_reduced, y_train)

# predict using the test data
y_pred = lr_clf.predict(X_test_vect_reduced)

vectorizer = NLTKVectorizer(stop_words=stop,      
                            max_df=0.5, min_df=10, max_features=10000)

# logistic Regression classifier
lr_clf = LogisticRegression(C=1.0, solver='newton-cg', multi_class='multinomial')

# create pipeline object
pipeline = Pipeline([
    ('vect', vectorizer),
    ('clf', lr_clf)
])

pipeline.fit(X_train, y_train)

# use the pipeline for predicting using test data
y_pred = pipeline.predict(X_test)

from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

lr_clf = LogisticRegression(C=1.0, solver='newton-cg', multi_class='multinomial')

# Naive Bayes classifier
nb_clf = MultinomialNB(alpha=0.01)

# SVM classifier
svm_clf = LinearSVC(C=1.0)

# Random Forest classifier
random_forest_clf = RandomForestClassifier(n_estimators=100, criterion='gini',
                                           max_depth=50, random_state=0)

# define the parameters list
parameters = {
    # vectorizer hyper-parameters
    'vect__ngram_range': [(1, 1), (1, 2),(1,3)],
    'vect__max_df': [0.5, 1.0],
    'vect__min_df': [1,2],
    'vect__max_features': [None,1000,5000],
    # classifiers
    'clf': [svm_clf]
}

# create grid search object, and use the pipeline as an estimator
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1)

# fit the grid search on the training data
grid_search.fit(X_train, y_train)

# get the list of optimal parameters
print(grid_search.best_params_)

vectorizer = NLTKVectorizer(max_df=1.0, min_df=1, ngram_range=(1, 1),
                            max_features=1000, stop_words=stop)

svm_clf = LinearSVC(C=1.0)

clf = CalibratedClassifierCV(base_estimator=svm_clf, cv=5, method='isotonic')

pipeline = Pipeline([
    ('vect', vectorizer),
    ('clf', clf)
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

from sklearn.metrics import classification_report      
y_true = ['fake','true']
y_pred = ['fake','true']
target_names = ['class 0', 'class 1']
print(classification_report(y_true, y_pred, target_names=target_names))