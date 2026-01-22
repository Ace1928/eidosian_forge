import math
import nltk.classify.util  # for accuracy & log_likelihood
from nltk.util import LazyMap
def binary_names_demo_features(name):
    features = {}
    features['alwayson'] = True
    features['startswith(vowel)'] = name[0].lower() in 'aeiouy'
    features['endswith(vowel)'] = name[-1].lower() in 'aeiouy'
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features['count(%s)' % letter] = name.lower().count(letter)
        features['has(%s)' % letter] = letter in name.lower()
        features['startswith(%s)' % letter] = letter == name[0].lower()
        features['endswith(%s)' % letter] = letter == name[-1].lower()
    return features