from functools import reduce
from nltk.parse.api import ParserI
from nltk.tree import ProbabilisticTree, Tree

        Print trace output indicating that a given production has been
        applied at a given location.

        :param production: The production that has been applied
        :type production: Production
        :param p: The probability of the tree produced by the production.
        :type p: float
        :param span: The span of the production
        :type span: tuple
        :rtype: None
        