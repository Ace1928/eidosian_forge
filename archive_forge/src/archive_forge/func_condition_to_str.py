from abc import ABCMeta, abstractmethod
from nltk import jsontags
def condition_to_str(feature, value):
    return 'the {} of {} is "{}"'.format(feature.PROPERTY_NAME, range_to_str(feature.positions), value)