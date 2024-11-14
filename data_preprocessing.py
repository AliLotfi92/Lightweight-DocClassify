import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from tqdm import tqdm

tqdm.pandas()
stemmer = PorterStemmer()
nltk.download('punkt')
nltk.download('stopwords')

nlp = spacy.load("en_core_web_sm")
df = pd.read_csv("data/df_file.csv")
stop_words = set(stopwords.words("english"))


def preprocessing(text):
    # removing \n
    text = text.replace("\n", "")
    # lowercasing as it does not have any 
    text = text.lower()
    text = ' '.join(text.split())
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    text = ''.join([char for char in text if not char.isdigit()])
    doc = nlp(text)
    cleaned_text = ' '.join([token.text for token in doc if token.ent_type_ != "PERSON"])
    stemmed_text = ' '.join([stemmer.stem(word) for word in cleaned_text.split()])
    word_tokens = word_tokenize(stemmed_text)

    filtered_sentence = [word for word in word_tokens if word.lower() not in stop_words]
    filtered_text = " ".join(filtered_sentence)

    return filtered_text

if __name__ == "__main__":
    df["processed_text"] = df["Text"].progress_apply(preprocessing)
    df.to_csv("data/df_file_processed.csv", index=False)

