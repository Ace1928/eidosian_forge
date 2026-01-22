import os
from itertools import chain
import nltk
from nltk.internals import Counter
from nltk.sem import drt, linearlogic
from nltk.sem.logic import (
from nltk.tag import BigramTagger, RegexpTagger, TrigramTagger, UnigramTagger
def lambda_abstract(self, other):
    assert isinstance(other, GlueFormula)
    assert isinstance(other.meaning, AbstractVariableExpression)
    return self.__class__(self.make_LambdaExpression(other.meaning.variable, self.meaning), linearlogic.ImpExpression(other.glue, self.glue))