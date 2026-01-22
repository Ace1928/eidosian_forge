import functools
import re
import nltk.tree
def tgrep_compile(tgrep_string):
    """
    Parses (and tokenizes, if necessary) a TGrep search string into a
    lambda function.
    """
    parser = _build_tgrep_parser(True)
    if isinstance(tgrep_string, bytes):
        tgrep_string = tgrep_string.decode()
    return list(parser.parseString(tgrep_string, parseAll=True))[0]