from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
from data_preprocessing import preprocessing

app = Flask(__name__)

# loading artifacts
with open('resources/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('resources/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)
with open('resources/label_encoder.pkl', 'rb') as file:
    loaded_encoder = pickle.load(file)


# creating api post 
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    texts = data.get('texts', [])
    
    if not texts:
        return jsonify({'error': 'there was no message'}), 400

    # preprocessing and vectorizing
    transformed_texts = [preprocessing(text) for text in texts]
    X = tfidf_vectorizer.transform(transformed_texts).toarray()
    print(X.shape)
    

    # list to store predictions and probabilities for each classifier
    ind_predictions = []
    ind_probabilities = []

    # prediction and probabilies for each classifier in stacking classifier
    for clf in model.estimators_:
        clf_predictions = clf.predict(X)
        clf_probabilities = clf.predict_proba(X)
        ind_predictions.append(clf_predictions)
        ind_probabilities.append(clf_probabilities)
        
    # reshaping for better accessibility
    ind_predictions = np.array(ind_predictions).T 
    ind_probabilities = np.array(ind_probabilities).transpose(1, 0, 2)

    final_predictions = model.predict(X)
    final_classes = loaded_encoder.inverse_transform(final_predictions.tolist())
    final_probabilities = model.predict_proba(X)
    response = {
        'predictions': final_predictions.tolist(),
        'probabilities': final_probabilities.tolist(),
        'predictions_str': final_classes.tolist(),
        'classifiers_predictions': ind_predictions.tolist(),
        'classifiers_probabilities': ind_probabilities.tolist()
    }

    return jsonify(response), 200

if __name__ == '__main__':
    app.run(debug=True, port=5001)