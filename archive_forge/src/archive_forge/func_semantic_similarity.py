import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk, tree2conlltags
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
def semantic_similarity(word1: str, word2: str) -> float:
    """
    Calculate semantic similarity between two words using WordNet.
    """
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)
    if synsets1 and synsets2:
        return max((s1.path_similarity(s2) or 0 for s1 in synsets1 for s2 in synsets2))
    return 0.0