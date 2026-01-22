import math
import nltk.classify.util  # for accuracy & log_likelihood
from nltk.util import LazyMap
def attested_labels(tokens):
    """
    :return: A list of all labels that are attested in the given list
        of tokens.
    :rtype: list of (immutable)
    :param tokens: The list of classified tokens from which to extract
        labels.  A classified token has the form ``(token, label)``.
    :type tokens: list
    """
    return tuple({label for tok, label in tokens})