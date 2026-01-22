import re
from nltk.tokenize.api import TokenizerI
from nltk.tokenize.util import regexp_span_tokenize
def regexp_tokenize(text, pattern, gaps=False, discard_empty=True, flags=re.UNICODE | re.MULTILINE | re.DOTALL):
    """
    Return a tokenized copy of *text*.  See :class:`.RegexpTokenizer`
    for descriptions of the arguments.
    """
    tokenizer = RegexpTokenizer(pattern, gaps, discard_empty, flags)
    return tokenizer.tokenize(text)