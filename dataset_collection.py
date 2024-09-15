# dataset collection from https://research.google.com/audioset/dataset/index.html using yt_dlp

import pandas as pd
import os
from yt_dlp import YoutubeDL
from yt_dlp.utils import download_range_func


def download_audio(yt_id, start_sec, end_sec, class_name):
    """
    download audio from a youtube video id from start_sec to end_sec
    save the file to /datasets/{class_name}/ directory
    """
    video_url = f'https://www.youtube.com/watch?v={yt_id}'
    ydl_opts = {
        # "verbose": True,
        "download_ranges": download_range_func(None, [(start_sec, end_sec)]),
        "force_keyframes_at_cuts": True,
        "format": "bestaudio/best",
        "outtmpl": os.path.join("datasets", class_name, "%(id)s.%(ext)s"),
        "postprocessors": [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }]
    }

    with YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([video_url])
            print(f"Successfully downloaded video segment from {start_sec} to {end_sec}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")


def check_class(string, class_codes):
    """
    given a string (like /m/068hy,/m/07q6cd_,/m/0bt9lr,/m/0jbk),
    check if a given substring (like /m/068hy) exists within the string
    if yes return True, if no return False
    """
    for code in class_codes:
        if code in string:
            return True
        
    return False


# collecting the raw audio files for the dataset to be processed later
if __name__ == "__main__":
    class_map = {
        "music": ["/m/04rlf"],
        "speech": ["/m/09x0r"],
        "noise": ["/m/0jbk", "/t/dd00129", "/m/07yv9"]
    }
    classes = [
        # "music", 
        "speech", 
        # "noise"
    ]
    max_n_per_class = 500

    df = pd.read_csv("./eval_segments.csv", sep=", ")

    for class_name in classes:
        # create a subset of df of the same class
        df_class = df[df["classes"].apply(lambda x: check_class(x, class_map[class_name]))]

        # reset index (so the indexes are 0, 1, 2, etc)
        df_class = df_class.reset_index(drop=True)

        # loop for each row in the subset df
        for i in range(max_n_per_class):
            if i > len(df_class):
                break

            yt_id = df_class.iloc[i]["yt_id"]
            start_sec = df_class.iloc[i]["start"]
            end_sec = df_class.iloc[i]["end"]

            # if file is already downloaded, skip
            if os.path.exists(os.path.join("datasets", class_name, f"{yt_id}.wav")):
                print(f"{yt_id} is already downloaded, skipping")
                continue

            print(f"downloading {yt_id} of class {class_name}")
            download_audio(
                yt_id=yt_id,
                start_sec=start_sec,
                end_sec=end_sec,
                class_name=class_name
            )
