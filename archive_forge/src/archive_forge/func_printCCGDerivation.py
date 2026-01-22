import itertools
from nltk.ccg.combinator import *
from nltk.ccg.combinator import (
from nltk.ccg.lexicon import Token, fromstring
from nltk.ccg.logic import *
from nltk.parse import ParserI
from nltk.parse.chart import AbstractChartRule, Chart, EdgeI
from nltk.sem.logic import *
from nltk.tree import Tree
def printCCGDerivation(tree):
    leafcats = tree.pos()
    leafstr = ''
    catstr = ''
    for leaf, cat in leafcats:
        str_cat = '%s' % cat
        nextlen = 2 + max(len(leaf), len(str_cat))
        lcatlen = (nextlen - len(str_cat)) // 2
        rcatlen = lcatlen + (nextlen - len(str_cat)) % 2
        catstr += ' ' * lcatlen + str_cat + ' ' * rcatlen
        lleaflen = (nextlen - len(leaf)) // 2
        rleaflen = lleaflen + (nextlen - len(leaf)) % 2
        leafstr += ' ' * lleaflen + leaf + ' ' * rleaflen
    print(leafstr.rstrip())
    print(catstr.rstrip())
    printCCGTree(0, tree)