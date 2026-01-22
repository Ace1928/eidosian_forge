import itertools
import re
import warnings
from functools import total_ordering
from nltk.grammar import PCFG, is_nonterminal, is_terminal
from nltk.internals import raise_unorderable_types
from nltk.parse.api import ParserI
from nltk.tree import Tree
from nltk.util import OrderedDict
def demo_grammar():
    from nltk.grammar import CFG
    return CFG.fromstring('\nS  -> NP VP\nPP -> "with" NP\nNP -> NP PP\nVP -> VP PP\nVP -> Verb NP\nVP -> Verb\nNP -> Det Noun\nNP -> "John"\nNP -> "I"\nDet -> "the"\nDet -> "my"\nDet -> "a"\nNoun -> "dog"\nNoun -> "cookie"\nVerb -> "ate"\nVerb -> "saw"\nPrep -> "with"\nPrep -> "under"\n')