import operator
import warnings
def interval_distance(label1, label2):
    """Krippendorff's interval distance metric

    >>> from nltk.metrics import interval_distance
    >>> interval_distance(1,10)
    81

    Krippendorff 1980, Content Analysis: An Introduction to its Methodology
    """
    try:
        return pow(label1 - label2, 2)
    except:
        print('non-numeric labels not supported with interval distance')