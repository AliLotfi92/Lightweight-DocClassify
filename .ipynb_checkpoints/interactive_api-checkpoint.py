import streamlit as st
import pickle
import numpy as np
from data_preprocessing import preprocessing

# get/cache artifacts
@st.cache_resource
def load_model():
    with open('resources/model.pkl', 'rb') as model_file:
        return pickle.load(model_file)
@st.cache_resource
def load_vectorizer():
    with open('resources/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        return pickle.load(vectorizer_file)


@st.cache_resource
def load_encoder():
    with open('resources/label_encoder.pkl', 'rb') as file:
        label_encoder= pickle.load(file)
        return label_encoder

model = load_model()
tfidf_vectorizer = load_vectorizer()
label_encoder = load_encoder()

st.title('Motorcycle or Auto Classifier')

# placeholder for message
message = st.text_area("Enter the message/document here:", height=100)

if st.button('Classify'):
    if message:
        #preprocess and vectorizing
        transformed_message = preprocessing(message)
        X = tfidf_vectorizer.transform([transformed_message]).toarray()

        # prediction for each classifier
        individual_predictions = []
        individual_probabilities = []
        for clf in model.estimators_:
            # getting prediction and probabilites
            clf_prediction = clf.predict(X)
            clf_probabilities = clf.predict_proba(X)
            individual_predictions.append(clf_prediction[0])
            individual_probabilities.append(clf_probabilities[0])

        final_prediction = model.predict(X)[0]
        final_probabilities = model.predict_proba(X)[0]

        #showing the results
        st.subheader('Final Prediction')
        st.write(f"Class: {final_prediction}")
        st.write("Probabilities:")
        for i, prob in enumerate(final_probabilities):
            st.write(f"Class {label_encoder.inverse_transform([i])}: {prob:.4f}")

        st.subheader('Each Classifier Predictions')
        # all classifiers
        classifiers_list = ["SVC", "NB", "RF", "ETC", "XGB"]
        for i, (pred, probs) in enumerate(zip(individual_predictions, individual_probabilities)):
            st.write(f"Classifier {classifiers_list[i]}:")
            st.write(f"  Prediction: {pred}")
            st.write("  Probabilities:")
            for j, prob in enumerate(probs):
                st.write(f"    Class {label_encoder.inverse_transform([j])}: {prob:.4f}")
            st.write("-"*50)
    else:
        st.warning('Please enter a message to classify.')