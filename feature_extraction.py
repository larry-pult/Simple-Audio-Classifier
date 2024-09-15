import librosa
import numpy as np
import pandas as pd
import os

N_MFCC_DEFAULT = 32


def get_duration(y, sr):
    return int(librosa.get_duration(y=y, sr=sr))


def get_duration_from_directory(directory):
    y, sr = librosa.load(directory)
    return get_duration(y, sr)


def extract_features(y, sr, n_mfcc=N_MFCC_DEFAULT):
    """
    extract all features (chroma, mfcc, spectral centroid, zero crossing rate) from a specified second

    then return a dictionary which contains the feature names as a key, and feature values as value
    """

    # the output dictionary
    features = {}

    # extract mean and stdev of chroma features, insert to output dictionary
    # also calculate the combined mean and stdev of all 12 chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_means = []
    for i in range(12):
        mean = np.mean(chroma[i])
        chroma_means.append(mean)
        features[f"chroma_{i}_mean"] = mean
        features[f"chroma_{i}_stdev"] = np.std(chroma[i])
    features["chroma_combined_mean"] = np.mean(chroma_means)
    features["chroma_combined_stdev"] = np.std(chroma_means)

    # extract mean and stdev of mfcc, insert to output dictionary
    # also calculate the combined mean and stdev of all mfccs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_means = []
    for i in range(n_mfcc):
        mean = np.mean(mfccs[i])
        mfcc_means.append(mean)
        features[f"mfcc_{i}_mean"] = mean
        features[f"mfcc_{i}_stdev"] = np.std(mfccs[i])
    features["mfcc_combined_mean"] = np.mean(mfcc_means)
    features["mfcc_combined_stdev"] = np.std(mfcc_means)

    # extract mean and stdev of spectral centroid, insert to output dictionary
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features["sc_mean"] = np.mean(spectral_centroid)
    features["sc_stdev"] = np.std(spectral_centroid)

    # extract mean and stdev of zero crossing rate, insert to output dictionary
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
    features["zcr_mean"] = np.mean(zero_crossing_rate)
    features["zcr_stdev"] = np.std(zero_crossing_rate)

    return features


def extract_features_from_directory(directory, start_sec=0, duration_sec=1, n_mfcc=N_MFCC_DEFAULT):
    # load file from directory
    y, sr = librosa.load(
        directory, 
        offset=start_sec,
        duration=duration_sec
    )

    features = {
        "start_sec": start_sec,
        "end_sec": start_sec + duration_sec
    }

    # extract features
    features.update(extract_features(y, sr, n_mfcc=n_mfcc))

    return features


# extracting the features from the raw audio files, saving it to extracted_features.csv
if __name__ == "__main__":
    classes = os.listdir("datasets")
    df = []

    for class_name in classes:
        file_names = os.listdir(os.path.join("datasets", class_name))

        for i, file_name in enumerate(file_names):
            directory = os.path.join("datasets", class_name, file_name)
            duration = get_duration_from_directory(directory)

            for start_sec in range(duration - 1):
                try:
                    features = {
                        "file_name": file_name,
                        "class_name": class_name
                    }
                    features.update(
                        extract_features_from_directory(
                            directory=directory,
                            start_sec=start_sec,
                            n_mfcc=N_MFCC_DEFAULT
                        )
                    )
                    df.append(features)
                except Exception as e:
                    print(f"skipping due to error: {e}")
        
        print(f"finished extracting features of class {class_name}")
    
    df = pd.DataFrame(df)

    df.to_csv("./extracted_features.csv", index=False)