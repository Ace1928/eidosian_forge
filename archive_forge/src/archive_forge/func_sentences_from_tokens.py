import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
def sentences_from_tokens(self, tokens: Iterator[PunktToken]) -> Iterator[PunktToken]:
    """
        Given a sequence of tokens, generates lists of tokens, each list
        corresponding to a sentence.
        """
    tokens = iter(self._annotate_tokens((self._Token(t) for t in tokens)))
    sentence = []
    for aug_tok in tokens:
        sentence.append(aug_tok.tok)
        if aug_tok.sentbreak:
            yield sentence
            sentence = []
    if sentence:
        yield sentence