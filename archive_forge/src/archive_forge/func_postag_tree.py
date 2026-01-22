import os
import pickle
import re
from xml.etree import ElementTree as ET
from nltk.tag import ClassifierBasedTagger, pos_tag
from nltk.chunk.api import ChunkParserI
from nltk.chunk.util import ChunkScore
from nltk.data import find
from nltk.tokenize import word_tokenize
from nltk.tree import Tree
def postag_tree(tree):
    words = tree.leaves()
    tag_iter = (pos for word, pos in pos_tag(words))
    newtree = Tree('S', [])
    for child in tree:
        if isinstance(child, Tree):
            newtree.append(Tree(child.label(), []))
            for subchild in child:
                newtree[-1].append((subchild, next(tag_iter)))
        else:
            newtree.append((child, next(tag_iter)))
    return newtree