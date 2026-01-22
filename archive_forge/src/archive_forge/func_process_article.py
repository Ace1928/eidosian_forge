import bz2
import logging
import multiprocessing
import re
import signal
from pickle import PicklingError
from xml.etree.ElementTree import iterparse
from gensim import utils
from gensim.corpora.dictionary import Dictionary
from gensim.corpora.textcorpus import TextCorpus
def process_article(args, tokenizer_func=tokenize, token_min_len=TOKEN_MIN_LEN, token_max_len=TOKEN_MAX_LEN, lower=True):
    """Parse a Wikipedia article, extract all tokens.

    Notes
    -----
    Set `tokenizer_func` (defaults is :func:`~gensim.corpora.wikicorpus.tokenize`) parameter for languages
    like Japanese or Thai to perform better tokenization.
    The `tokenizer_func` needs to take 4 parameters: (text: str, token_min_len: int, token_max_len: int, lower: bool).

    Parameters
    ----------
    args : (str, str, int)
        Article text, article title, page identificator.
    tokenizer_func : function
        Function for tokenization (defaults is :func:`~gensim.corpora.wikicorpus.tokenize`).
        Needs to have interface:
        tokenizer_func(text: str, token_min_len: int, token_max_len: int, lower: bool) -> list of str.
    token_min_len : int
        Minimal token length.
    token_max_len : int
        Maximal token length.
    lower : bool
         Convert article text to lower case?

    Returns
    -------
    (list of str, str, int)
        List of tokens from article, title and page id.

    """
    text, title, pageid = args
    text = filter_wiki(text)
    result = tokenizer_func(text, token_min_len, token_max_len, lower)
    return (result, title, pageid)