from itertools import chain, product
from typing import Callable, Iterable, List, Tuple
from nltk.corpus import WordNetCorpusReader, wordnet
from nltk.stem.api import StemmerI
from nltk.stem.porter import PorterStemmer
def wordnetsyn_match(hypothesis: Iterable[str], reference: Iterable[str], wordnet: WordNetCorpusReader=wordnet) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    """
    Matches each word in reference to a word in hypothesis if any synonym
    of a hypothesis word is the exact match to the reference word.

    :param hypothesis: pre-tokenized hypothesis
    :param reference: pre-tokenized reference
    :param wordnet: a wordnet corpus reader object (default nltk.corpus.wordnet)
    :return: list of mapped tuples
    """
    enum_hypothesis_list, enum_reference_list = _generate_enums(hypothesis, reference)
    return _enum_wordnetsyn_match(enum_hypothesis_list, enum_reference_list, wordnet=wordnet)