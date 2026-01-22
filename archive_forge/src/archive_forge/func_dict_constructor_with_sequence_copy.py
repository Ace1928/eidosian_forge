import ast
from hacking import core
import re
@core.flake8ext
def dict_constructor_with_sequence_copy(logical_line):
    """Should use a dict comprehension instead of a dict constructor.

    PEP-0274 introduced dict comprehension with performance enhancement
    and it also makes code more readable.

    Okay: lower_res = {k.lower(): v for k, v in res[1].items()}
    Okay: fool = dict(a='a', b='b')
    K008: lower_res = dict((k.lower(), v) for k, v in res[1].items())
    K008:     attrs = dict([(k, _from_json(v))
    K008: dict([[i,i] for i in range(3)])

    """
    MESSAGE = 'K008 Must use a dict comprehension instead of a dict constructor with a sequence of key-value pairs.'
    dict_constructor_with_sequence_re = re.compile('.*\\bdict\\((\\[)?(\\(|\\[)(?!\\{)')
    if dict_constructor_with_sequence_re.match(logical_line):
        yield (0, MESSAGE)