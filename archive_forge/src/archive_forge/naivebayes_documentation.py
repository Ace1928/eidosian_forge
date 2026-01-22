from collections import defaultdict
from nltk.classify.api import ClassifierI
from nltk.probability import DictionaryProbDist, ELEProbDist, FreqDist, sum_logs

        :param labeled_featuresets: A list of classified featuresets,
            i.e., a list of tuples ``(featureset, label)``.
        