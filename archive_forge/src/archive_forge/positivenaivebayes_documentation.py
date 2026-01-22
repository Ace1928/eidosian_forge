from collections import defaultdict
from nltk.classify.naivebayes import NaiveBayesClassifier
from nltk.probability import DictionaryProbDist, ELEProbDist, FreqDist

        :param positive_featuresets: An iterable of featuresets that are known as positive
            examples (i.e., their label is ``True``).

        :param unlabeled_featuresets: An iterable of featuresets whose label is unknown.

        :param positive_prob_prior: A prior estimate of the probability of the label
            ``True`` (default 0.5).
        