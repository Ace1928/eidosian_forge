from collections import Counter, defaultdict
from nltk import jsontags
from nltk.tag import TaggerI
from nltk.tbl import Feature, Template
def fntbl37():
    """
    Return 37 templates taken from the postagging task of the
    fntbl distribution https://www.cs.jhu.edu/~rflorian/fntbl/
    (37 is after excluding a handful which do not condition on Pos[0];
    fntbl can do that but the current nltk implementation cannot.)
    """
    return [Template(Word([0]), Word([1]), Word([2])), Template(Word([-1]), Word([0]), Word([1])), Template(Word([0]), Word([-1])), Template(Word([0]), Word([1])), Template(Word([0]), Word([2])), Template(Word([0]), Word([-2])), Template(Word([1, 2])), Template(Word([-2, -1])), Template(Word([1, 2, 3])), Template(Word([-3, -2, -1])), Template(Word([0]), Pos([2])), Template(Word([0]), Pos([-2])), Template(Word([0]), Pos([1])), Template(Word([0]), Pos([-1])), Template(Word([0])), Template(Word([-2])), Template(Word([2])), Template(Word([1])), Template(Word([-1])), Template(Pos([-1]), Pos([1])), Template(Pos([1]), Pos([2])), Template(Pos([-1]), Pos([-2])), Template(Pos([1])), Template(Pos([-1])), Template(Pos([-2])), Template(Pos([2])), Template(Pos([1, 2, 3])), Template(Pos([1, 2])), Template(Pos([-3, -2, -1])), Template(Pos([-2, -1])), Template(Pos([1]), Word([0]), Word([1])), Template(Pos([1]), Word([0]), Word([-1])), Template(Pos([-1]), Word([-1]), Word([0])), Template(Pos([-1]), Word([0]), Word([1])), Template(Pos([-2]), Pos([-1])), Template(Pos([1]), Pos([2])), Template(Pos([1]), Pos([2]), Word([1]))]