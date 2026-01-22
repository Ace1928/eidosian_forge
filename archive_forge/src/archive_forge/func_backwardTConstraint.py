from abc import ABCMeta, abstractmethod
from nltk.ccg.api import FunctionalCategory
def backwardTConstraint(left, right):
    arg = innermostFunction(left)
    return arg.dir().is_forward() and arg.res().is_primitive()