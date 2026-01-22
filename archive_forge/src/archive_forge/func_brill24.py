from collections import Counter, defaultdict
from nltk import jsontags
from nltk.tag import TaggerI
from nltk.tbl import Feature, Template
def brill24():
    """
    Return 24 templates of the seminal TBL paper, Brill (1995)
    """
    return [Template(Pos([-1])), Template(Pos([1])), Template(Pos([-2])), Template(Pos([2])), Template(Pos([-2, -1])), Template(Pos([1, 2])), Template(Pos([-3, -2, -1])), Template(Pos([1, 2, 3])), Template(Pos([-1]), Pos([1])), Template(Pos([-2]), Pos([-1])), Template(Pos([1]), Pos([2])), Template(Word([-1])), Template(Word([1])), Template(Word([-2])), Template(Word([2])), Template(Word([-2, -1])), Template(Word([1, 2])), Template(Word([-1, 0])), Template(Word([0, 1])), Template(Word([0])), Template(Word([-1]), Pos([-1])), Template(Word([1]), Pos([1])), Template(Word([0]), Word([-1]), Pos([-1])), Template(Word([0]), Word([1]), Pos([1]))]