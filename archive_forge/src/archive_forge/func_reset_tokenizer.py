import abc
import functools
import re
import tokenize
from tokenize import untokenize, TokenError
from io import StringIO
from IPython.core.splitinput import LineInfo
from IPython.utils import tokenutil
def reset_tokenizer(self):
    it = iter(self.buf)
    self.tokenizer = tokenutil.generate_tokens_catch_errors(it.__next__)