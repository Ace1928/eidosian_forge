import math
import nltk.classify.util  # for accuracy & log_likelihood
from nltk.util import LazyMap
def lazy_func(labeled_token):
    return (feature_func(labeled_token[0]), labeled_token[1])