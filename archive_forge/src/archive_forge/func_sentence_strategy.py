import re
import string
import hypothesis
import hypothesis.strategies
import pytest
import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import get_lang_class
@hypothesis.strategies.composite
def sentence_strategy(draw: hypothesis.strategies.DrawFn, max_n_words: int=4) -> str:
    """
    Composite strategy for fuzzily generating sentence with varying interpunctation.

    draw (hypothesis.strategies.DrawFn): Protocol for drawing function allowing to fuzzily pick from hypothesis'
                                         strategies.
    max_n_words (int): Max. number of words in generated sentence.
    RETURNS (str): Fuzzily generated sentence.
    """
    punctuation_and_space_regex = '|'.join([*[re.escape(p) for p in string.punctuation], '\\s'])
    sentence = [[draw(hypothesis.strategies.text(min_size=1)), draw(hypothesis.strategies.from_regex(punctuation_and_space_regex))] for _ in range(draw(hypothesis.strategies.integers(min_value=2, max_value=max_n_words)))]
    return ' '.join([token for token_pair in sentence for token in token_pair])