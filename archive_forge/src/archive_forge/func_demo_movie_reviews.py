import codecs
import csv
import json
import pickle
import random
import re
import sys
import time
from copy import deepcopy
import nltk
from nltk.corpus import CategorizedPlaintextCorpusReader
from nltk.data import load
from nltk.tokenize.casual import EMOTICON_RE
def demo_movie_reviews(trainer, n_instances=None, output=None):
    """
    Train classifier on all instances of the Movie Reviews dataset.
    The corpus has been preprocessed using the default sentence tokenizer and
    WordPunctTokenizer.
    Features are composed of:

    - most frequent unigrams

    :param trainer: `train` method of a classifier.
    :param n_instances: the number of total reviews that have to be used for
        training and testing. Reviews will be equally split between positive and
        negative.
    :param output: the output file where results have to be reported.
    """
    from nltk.corpus import movie_reviews
    from nltk.sentiment import SentimentAnalyzer
    if n_instances is not None:
        n_instances = int(n_instances / 2)
    pos_docs = [(list(movie_reviews.words(pos_id)), 'pos') for pos_id in movie_reviews.fileids('pos')[:n_instances]]
    neg_docs = [(list(movie_reviews.words(neg_id)), 'neg') for neg_id in movie_reviews.fileids('neg')[:n_instances]]
    train_pos_docs, test_pos_docs = split_train_test(pos_docs)
    train_neg_docs, test_neg_docs = split_train_test(neg_docs)
    training_docs = train_pos_docs + train_neg_docs
    testing_docs = test_pos_docs + test_neg_docs
    sentim_analyzer = SentimentAnalyzer()
    all_words = sentim_analyzer.all_words(training_docs)
    unigram_feats = sentim_analyzer.unigram_word_feats(all_words, min_freq=4)
    sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)
    training_set = sentim_analyzer.apply_features(training_docs)
    test_set = sentim_analyzer.apply_features(testing_docs)
    classifier = sentim_analyzer.train(trainer, training_set)
    try:
        classifier.show_most_informative_features()
    except AttributeError:
        print('Your classifier does not provide a show_most_informative_features() method.')
    results = sentim_analyzer.evaluate(test_set)
    if output:
        extr = [f.__name__ for f in sentim_analyzer.feat_extractors]
        output_markdown(output, Dataset='Movie_reviews', Classifier=type(classifier).__name__, Tokenizer='WordPunctTokenizer', Feats=extr, Results=results, Instances=n_instances)