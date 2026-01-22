import abc
import functools
import re
import tokenize
from tokenize import untokenize, TokenError
from io import StringIO
from IPython.core.splitinput import LineInfo
from IPython.utils import tokenutil
@CoroutineInputTransformer.wrap
def ipy_prompt():
    """Strip IPython's In [1]:/...: prompts."""
    prompt_re = re.compile('^(In \\[\\d+\\]: |\\s*\\.{3,}: ?)')
    turnoff_re = re.compile('^%%')
    return _strip_prompts(prompt_re, turnoff_re=turnoff_re)