import operator
from collections import defaultdict
from functools import reduce
from nltk.inference.api import BaseProverCommand, Prover
from nltk.sem import skolemize
from nltk.sem.logic import (

        :param other: ``BindingDict`` The dict with which to combine self
        :return: ``BindingDict`` A new dict containing all the elements of both parameters
        :raise BindingException: If the parameter dictionaries are not consistent with each other
        