import subprocess
import warnings
from collections import defaultdict
from itertools import chain
from pprint import pformat
from nltk.internals import find_binary
from nltk.tree import Tree
def extract_7_cells(cells, index):
    line_index, word, lemma, tag, _, head, rel = cells
    try:
        index = int(line_index)
    except ValueError:
        pass
    return (index, word, lemma, tag, tag, '', head, rel)