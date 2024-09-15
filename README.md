## Simple Audio Classifier

### How To Set Up
1. Install Python 3.12 (maybe you can try with older versions, not sure)
2. Set up a new virtual environment (maybe you can use an existing one, not sure)
3. Install the packages `from requirements.txt`
4. Import and use the functions from `prediction.py` such as `extract_audio_from_video_directory()` or `predict_class_from_directory()`

### Description of Files
- `dataset_collection.py`: collects raw audio files from Google Audio Dataset
- `feature_extraction.py`: takes in an audio file, extract its features such as mfcc and chroma, save it to `extracted_features.csv` or use for prediction
- `model_training.ipynb`: trains an xgboost model with the `extracted_features.csv` as training data, save trained model into `model_xgboost.json`
- `prediction.py`: takes in an audio file directory or raw audio data, extracts features using `feature_extraction.py`, gets prediction using the saved model `model_xgboost.json`

### Approach
#### Dataset Collection
- collecting data from Google Audio Dataset: `https://research.google.com/audioset/dataset/index.html`
  - why? because it is highly varied which will prevent overfitting
- it only provides youtube links instead of the raw data, available in `eval_segments.csv`
- so I have to download the audio files myself, automate it using `yt_dlp` library
  - will get blocked if downloading too much, just like scraping
#### Machine Learning
- Each audio is split into 1-second segments
- Extract features by calculating the mean and standard deviation of each features within each segment
  - e.g. chroma_1_mean, mfcc_31_stdev
  - also calculate the combined mean and stdev for all chroma and MFCC channels
- Classes: `music`, `speech`, `noise`
- Features used: chroma, MFCC (32 coefs), zero crossing rate, spectral centroid
  - MFCC and chroma have high impact
  - ZCR and SC very small impact
  - why use all? because it increases the accuracy, and they're cheap anyway
- Use XGBoost for model, a model with very strong predicting power for tabular data
- 90-10 train-test-split for evaluation

### Results
#### Dataset Collection
- 1500 raw audio files with about 5-10 seconds of audio
    - 500 for each class
- 13000 rows of machine learning data
- 90 features/columns for prediction
#### Machine Learning
- Train Accuracy: 0.91
- Test Accuracy: 0.65

### Future Work
- Get more training data, which should lessen the gap between train acc and test acc
- Experiment with more feature engineering (e.g. more statistical functions, chroma transposition)
- Experiment with smaller sampling windows
- Hyperparameter tuning? probably not, according to my  personal experience the impact should be very small, probably the 10th thing I would do
- Add more classes (e.g. silence)