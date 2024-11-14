import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle

df = pd.read_csv("data/df_file_processed.csv")

mapper = {"Politics": 0, "Sport": 1, "Technology": 2, "Entertainment":3, "Business":4}
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(list(mapper.keys()))


if __name__ == "__main__":
    print("Creating training and test datasets")
    df['label_class'] = label_encoder.inverse_transform(df['Label'])

    df = df.sample(frac=1).reset_index(drop=True)
    split_index = int(0.8 * len(df))
    df_train = df.iloc[:split_index]
    df_test = df.iloc[split_index:]
    # save string to int encoder
    with open('resources/label_encoder.pkl', 'wb') as file:
        pickle.dump(label_encoder, file)
    # save test and train data
    df_train.to_csv("data/train_dataset.csv", index=False)
    df_test.to_csv("data/test_dataset.csv", index=False)
    print("Done!")
        