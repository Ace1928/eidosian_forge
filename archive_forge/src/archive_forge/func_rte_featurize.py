from nltk.classify.maxent import MaxentClassifier
from nltk.classify.util import accuracy
from nltk.tokenize import RegexpTokenizer
def rte_featurize(rte_pairs):
    return [(rte_features(pair), pair.value) for pair in rte_pairs]