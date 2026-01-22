from re import finditer
from xml.sax.saxutils import escape, unescape
def spans_to_relative(spans):
    """
    Return a sequence of relative spans, given a sequence of spans.

        >>> from nltk.tokenize import WhitespaceTokenizer
        >>> from nltk.tokenize.util import spans_to_relative
        >>> s = '''Good muffins cost $3.88\\nin New York.  Please buy me
        ... two of them.\\n\\nThanks.'''
        >>> list(spans_to_relative(WhitespaceTokenizer().span_tokenize(s))) # doctest: +NORMALIZE_WHITESPACE
        [(0, 4), (1, 7), (1, 4), (1, 5), (1, 2), (1, 3), (1, 5), (2, 6),
        (1, 3), (1, 2), (1, 3), (1, 2), (1, 5), (2, 7)]

    :param spans: a sequence of (start, end) offsets of the tokens
    :type spans: iter(tuple(int, int))
    :rtype: iter(tuple(int, int))
    """
    prev = 0
    for left, right in spans:
        yield (left - prev, right - left)
        prev = right