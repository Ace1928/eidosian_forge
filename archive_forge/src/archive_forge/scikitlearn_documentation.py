from nltk.classify.api import ClassifierI
from nltk.probability import DictionaryProbDist

        Train (fit) the scikit-learn estimator.

        :param labeled_featuresets: A list of ``(featureset, label)``
            where each ``featureset`` is a dict mapping strings to either
            numbers, booleans or strings.
        