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
def demo_subjectivity(trainer, save_analyzer=False, n_instances=None, output=None):
    """
    Train and test a classifier on instances of the Subjective Dataset by Pang and
    Lee. The dataset is made of 5000 subjective and 5000 objective sentences.
    All tokens (words and punctuation marks) are separated by a whitespace, so
    we use the basic WhitespaceTokenizer to parse the data.

    :param trainer: `train` method of a classifier.
    :param save_analyzer: if `True`, store the SentimentAnalyzer in a pickle file.
    :param n_instances: the number of total sentences that have to be used for
        training and testing. Sentences will be equally split between positive
        and negative.
    :param output: the output file where results have to be reported.
    """
    from nltk.corpus import subjectivity
    from nltk.sentiment import SentimentAnalyzer
    if n_instances is not None:
        n_instances = int(n_instances / 2)
    subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:n_instances]]
    obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj')[:n_instances]]
    train_subj_docs, test_subj_docs = split_train_test(subj_docs)
    train_obj_docs, test_obj_docs = split_train_test(obj_docs)
    training_docs = train_subj_docs + train_obj_docs
    testing_docs = test_subj_docs + test_obj_docs
    sentim_analyzer = SentimentAnalyzer()
    all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in training_docs])
    unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
    sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)
    training_set = sentim_analyzer.apply_features(training_docs)
    test_set = sentim_analyzer.apply_features(testing_docs)
    classifier = sentim_analyzer.train(trainer, training_set)
    try:
        classifier.show_most_informative_features()
    except AttributeError:
        print('Your classifier does not provide a show_most_informative_features() method.')
    results = sentim_analyzer.evaluate(test_set)
    if save_analyzer == True:
        sentim_analyzer.save_file(sentim_analyzer, 'sa_subjectivity.pickle')
    if output:
        extr = [f.__name__ for f in sentim_analyzer.feat_extractors]
        output_markdown(output, Dataset='subjectivity', Classifier=type(classifier).__name__, Tokenizer='WhitespaceTokenizer', Feats=extr, Instances=n_instances, Results=results)
    return sentim_analyzer