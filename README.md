# stumbleupon
Kaggle submission for a Stumbleupon Evergreen classification challenge, good for 30th place out of 625 entries https://www.kaggle.com/c/stumbleupon

Competition's goal is to predict whether a given website is viral over long periods of time (Evergreen).  My approach focuses solely on the text of the articles as an exercise to practice NLP techniques. It is likely possible to further improve this score using additional information provided in the dataset.

To train: clone the repository, `pip install -r requiements.txt` for the necessary libraries, then type `python main.py` in the terminal to train the model. This will produce the `submission_final.csv` which can be scored by sending the .csv to https://www.kaggle.com/c/stumbleupon/submissions under "Make a Submission"

Model uses NLP techniques of tf-idf with word stemming, Latent Semantic Analysis for feature selection, then is trained via propritary ModelEnsemble class, which uses Stacked Generalization to weigh the model importances of Logistic Regression and Gradient Boosted classifiers.  The ModelEnsemble takes advantage of different decision bounds from different algorithms in an attempt to maximize against scoring metric, AUC.
