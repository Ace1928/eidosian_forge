from typing import List, Tuple
import pytest
from nltk.tokenize import (
def test_tweet_tokenizer(self):
    """
        Test TweetTokenizer using words with special and accented characters.
        """
    tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
    s9 = "@myke: Let's test these words: resumé España München français"
    tokens = tokenizer.tokenize(s9)
    expected = [':', "Let's", 'test', 'these', 'words', ':', 'resumé', 'España', 'München', 'français']
    assert tokens == expected