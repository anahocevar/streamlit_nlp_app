import json
import os
import pickle
import praw
import numpy as np
from praw.models import MoreComments
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from joblib import dump

def get_comments(word, limit=100):
    """Get the top level comments for a given number of hot submissions."""
    comments = []
    subreddit = reddit.subreddit(word)
    
    for submission_id in subreddit.hot(limit=limit):
        submission = reddit.submission(id=submission_id)
        
        for top_level_comment in submission.comments:
            if isinstance(top_level_comment, MoreComments):
                continue
            comments.append(top_level_comment.body)

    return comments

def fetch_corpus(word):
    """Return a list of Reddit comments, either by loading a file or fetching them from Reddit."""
    file_name = f'data/{word}.pkl'

    # Load file if it exists
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            comments = pickle.load(f)
    else:
        comments = get_comments(word)
        
    return comments

def find_top_words(model, N):
    """Return a list of two dictionaries, mapping top words and their weight for the class."""
    
    # Calculate log probability ratio for each word in our vocabulary
    log_prob_ratio = model[-1].feature_log_prob_[0, :] - model[-1].feature_log_prob_[1, :]
    ind_sorted = np.argsort(log_prob_ratio)
    
    # Dictionary mapping word to their weight of importance
    index_to_word = model[0].get_feature_names()
    top_words_class_0 = {index_to_word[i]: abs(log_prob_ratio[i]) for i in ind_sorted[:N]}
    top_words_class_1 = {index_to_word[i]: abs(log_prob_ratio[i]) for i in ind_sorted[-1:-N-1:-1]}

    return [top_words_class_0, top_words_class_1]


if __name__=='__main__':
    # Create reddit instance if credential file exists
    if os.path.exists('secrets/reddit_secrets.json'):
        with open('secrets/reddit_secrets.json', 'r') as f:
            secrets = json.load(f)

        reddit = praw.Reddit(client_id=secrets['client_id'],
                             client_secret=secrets['client_secret'],
                             user_agent=secrets['user_agent'])
    
    # Generate the corpus
    animal_comments = fetch_corpus('ballpython')
    language_comments = fetch_corpus('python')
    corpus = animal_comments + language_comments 

    # Create a label for each document
    labels = ['animal'] * len(animal_comments) + ['language'] * len(language_comments)

    # Split data into train-test
    X_train, X_test, y_train, y_test = train_test_split(
        corpus, labels, test_size=0.2, random_state=42)

    # Construct pipeline
    vectorizer = CountVectorizer()
    clf = MultinomialNB()
    pipe = Pipeline([('vectorizer', vectorizer), ('classifier', clf)])

    # Train and save model
    pipe.fit(X_train, y_train)
    dump(pipe, 'models/classifier.joblib') 
    print('Model was successfully saved.')

    top_words = find_top_words(pipe, 50)
    with open('models/top_words.pkl', 'wb') as f:
        pickle.dump(top_words, f)
    print('Top words were successfully saved.')