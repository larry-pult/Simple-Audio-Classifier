import pandas as pd
import librosa
import os
from xgboost import XGBClassifier
from scipy.stats import mode
from audioread import NoBackendError
from moviepy.editor import VideoFileClip
import feature_extraction  # import script from current folder

XGBOOST = XGBClassifier()
XGBOOST.load_model("./model_xgboost.json")

CLASS_MAPPINGS = {
    0: "music",
    1: "speech",
    2: "noise"
}


def predict_class(y, sr):
    """
    given the raw audio data and sampling rate,
    extract the features for each second and predict the classes for each second
    return the most frequent predicted class and the predictions per second
    """
    duration = feature_extraction.get_duration(y=y, sr=sr)
    # store the extracted features
    features = []

    # get features for each second
    for start_sec in range(int(duration)):
        # get only the audio subset of y from start_sec until start_sec+1
        start_index = sr*start_sec
        end_index = start_index + sr
        y_subset = y[start_index:end_index]

        # extract the predictions and then append to the results
        features.append(
            feature_extraction.extract_features(
                y=y_subset, 
                sr=sr,
                n_mfcc=32
            )
        )
    
    # convert the features into a dataframe
    features = pd.DataFrame(features)

    # get predictions
    y_pred = XGBOOST.predict(features)

    # get most frequent predictions
    class_mode = mode(y_pred).mode

    # get class names (music, speech, noise) from ordinal predictions (0, 1, 2)
    y_pred = [CLASS_MAPPINGS[class_code] for class_code in y_pred]
    class_mode = CLASS_MAPPINGS[class_mode]

    return {
        "most_frequent_prediction": class_mode,
        "predictions_per_second": y_pred
    }


def predict_class_from_directory(directory):
    """
    takes in the directory of an audio file,
    then get the predictions
    """
    # try loading the file and predict class
    try:
        y, sr = librosa.load(directory)
        return predict_class(y, sr)
    # this error is raised of the file being loaded is incompatible
    except NoBackendError:
        raise ValueError("file format incompatible")
    

def extract_audio_from_video_directory(source_dir, target_dir):
    """
    takes in a video file from source_dir
    then saves the audio into target_dir
    """
    # try loading the video then extract the audio into the specified directory
    try:
        video = VideoFileClip(source_dir)
        audio = video.audio
        audio.write_audiofile(target_dir)
    # this error is raised if the video file format is incompatible
    except OSError:
        raise ValueError("file format incompatible")

    
# example usage (given ./cache/example_video.mp4, extract audio and predict class)
if __name__ == "__main__":
    video_dir = os.path.join("cache", "example_video.mp4")
    target_audio_dir = os.path.join("cache", "example_video.wav")

    if not os.path.exists(target_audio_dir):
        extract_audio_from_video_directory(video_dir, target_audio_dir)
    else:
        print("audio file already exists in cache, skipping")
    
    prediction_results = predict_class_from_directory(target_audio_dir)

    print("-"*100)
    print(f"Prediction Results From {video_dir}:")
    print(f"most frequent prediction: {prediction_results["most_frequent_prediction"]}")
    print(f"predictions per second: {prediction_results["predictions_per_second"]}")