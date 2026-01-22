from nltk.inference.api import BaseProverCommand, Prover
from nltk.internals import Counter
from nltk.sem.logic import (
def put_all(self, expressions):
    for expression in expressions:
        self.put(expression)