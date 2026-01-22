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
def demo_vader_tweets(n_instances=None, output=None):
    """
    Classify 10000 positive and negative tweets using Vader approach.

    :param n_instances: the number of total tweets that have to be classified.
    :param output: the output file where results have to be reported.
    """
    from collections import defaultdict
    from nltk.corpus import twitter_samples
    from nltk.metrics import accuracy as eval_accuracy
    from nltk.metrics import f_measure as eval_f_measure
    from nltk.metrics import precision as eval_precision
    from nltk.metrics import recall as eval_recall
    from nltk.sentiment import SentimentIntensityAnalyzer
    if n_instances is not None:
        n_instances = int(n_instances / 2)
    fields = ['id', 'text']
    positive_json = twitter_samples.abspath('positive_tweets.json')
    positive_csv = 'positive_tweets.csv'
    json2csv_preprocess(positive_json, positive_csv, fields, strip_off_emoticons=False, limit=n_instances)
    negative_json = twitter_samples.abspath('negative_tweets.json')
    negative_csv = 'negative_tweets.csv'
    json2csv_preprocess(negative_json, negative_csv, fields, strip_off_emoticons=False, limit=n_instances)
    pos_docs = parse_tweets_set(positive_csv, label='pos')
    neg_docs = parse_tweets_set(negative_csv, label='neg')
    train_pos_docs, test_pos_docs = split_train_test(pos_docs)
    train_neg_docs, test_neg_docs = split_train_test(neg_docs)
    training_tweets = train_pos_docs + train_neg_docs
    testing_tweets = test_pos_docs + test_neg_docs
    vader_analyzer = SentimentIntensityAnalyzer()
    gold_results = defaultdict(set)
    test_results = defaultdict(set)
    acc_gold_results = []
    acc_test_results = []
    labels = set()
    num = 0
    for i, (text, label) in enumerate(testing_tweets):
        labels.add(label)
        gold_results[label].add(i)
        acc_gold_results.append(label)
        score = vader_analyzer.polarity_scores(text)['compound']
        if score > 0:
            observed = 'pos'
        else:
            observed = 'neg'
        num += 1
        acc_test_results.append(observed)
        test_results[observed].add(i)
    metrics_results = {}
    for label in labels:
        accuracy_score = eval_accuracy(acc_gold_results, acc_test_results)
        metrics_results['Accuracy'] = accuracy_score
        precision_score = eval_precision(gold_results[label], test_results[label])
        metrics_results[f'Precision [{label}]'] = precision_score
        recall_score = eval_recall(gold_results[label], test_results[label])
        metrics_results[f'Recall [{label}]'] = recall_score
        f_measure_score = eval_f_measure(gold_results[label], test_results[label])
        metrics_results[f'F-measure [{label}]'] = f_measure_score
    for result in sorted(metrics_results):
        print(f'{result}: {metrics_results[result]}')
    if output:
        output_markdown(output, Approach='Vader', Dataset='labeled_tweets', Instances=n_instances, Results=metrics_results)