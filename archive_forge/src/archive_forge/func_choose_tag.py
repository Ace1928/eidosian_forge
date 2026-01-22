import ast
import re
from abc import abstractmethod
from typing import List, Optional, Tuple
from nltk import jsontags
from nltk.classify import NaiveBayesClassifier
from nltk.probability import ConditionalFreqDist
from nltk.tag.api import FeaturesetTaggerI, TaggerI
def choose_tag(self, tokens, index, history):
    featureset = self.feature_detector(tokens, index, history)
    if self._cutoff_prob is None:
        return self._classifier.classify(featureset)
    pdist = self._classifier.prob_classify(featureset)
    tag = pdist.max()
    return tag if pdist.prob(tag) >= self._cutoff_prob else None