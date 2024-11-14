import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import string
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def train(train_dataset, processing_function):
    """
    the main training loop, 
    
    inputs:
        train_dataset: The training dataset in pd.DataFrame format .
        processing_function: The function for the message transformation.
    
    gives:
        results, trained model, and tfidf vectorizer.
    """
    # process the message & vectorzing the message
    tfidf_vectorizer = TfidfVectorizer(max_features=3000)
    X_train = tfidf_vectorizer.fit_transform(train_dataset['processed_text']).toarray()
    y_train = train_dataset['label_num'].values
    
    # 5 best performing classifiers tested individually for stacking classifier
    classifiers = {
        'svc': SVC(kernel="sigmoid", gamma=1.0, probability=True),
        'nb': MultinomialNB(),
        'rf': RandomForestClassifier(n_estimators=50, random_state=2),
        'etc': ExtraTreesClassifier(n_estimators=50, random_state=2),
        'xgb': XGBClassifier(n_estimators=50, random_state=2),
    }
    # stacking classifier get the probability each classifier above and then these would be input to a linear classifier to decide
    # the stacking classifier in general perfrom better than soft/hard voting. 
    # the final decision layer is logistig regression
    
    model = StackingClassifier(
        estimators=list(classifiers.items()),
        final_estimator=LogisticRegression(),
    )
    
    print("Training the model...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)


    # evaluation on training dataset
    results = {
        'accuracy': accuracy_score(y_train, y_pred),
        'precision': precision_score(y_train, y_pred, average='weighted'),
        'recall': recall_score(y_train, y_pred, average='weighted'),
        'f1_score': f1_score(y_train, y_pred, average='weighted'),
    }
    
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(classification_report(y_train, y_pred))
    print("-" * 30)
    
    return results, model, tfidf_vectorizer


def evaluate(model, tfidf_vectorizer, processing_function, test_dataset):
    """
    function to evaluate the performance on the test dataset
    inputs:
        model: model,
        tfidf_vectorizer: tfid vectorizer used for transformation.
        processing_function: prerpocessing function.
        test_dataset: The test dataset, ind pd.DataFRame format
    
    gives:
        results in dic format
    """
    test_dataset['transformed_message'] = test_dataset['message'].apply(processing_function)
    X_test = tfidf_vectorizer.transform(test_dataset['transformed_message']).toarray()
    y_test = test_dataset['label'].values
    y_pred = model.predict(X_test)
    
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
    }
    
    print("Test Results")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(classification_report(y_test, y_pred))
    print("-" * 30)
    
    return results


def main():
    # training and testing data
    df_train = pd.read_csv("data/train_dataset.csv")
    df_test = pd.read_csv("data/test_dataset.csv")

    # training & evaluating
    results, model, tfidf_vectorizer = train(df_train, preprocess_text)
    test_results = evaluate(model, tfidf_vectorizer, preprocess_text, df_test)
    
    # saving the model, and vectorizing:
    with open('resources/model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    with open('resources/tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
        pickle.dump(tfidf_vectorizer, vectorizer_file)
    print("model is saved")

if __name__ == "__main__":
    main()

    