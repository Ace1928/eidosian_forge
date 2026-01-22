import re
import regex
from nltk.chunk.api import ChunkParserI
from nltk.tree import Tree
def tag_pattern2re_pattern(tag_pattern):
    """
    Convert a tag pattern to a regular expression pattern.  A "tag
    pattern" is a modified version of a regular expression, designed
    for matching sequences of tags.  The differences between regular
    expression patterns and tag patterns are:

        - In tag patterns, ``'<'`` and ``'>'`` act as parentheses; so
          ``'<NN>+'`` matches one or more repetitions of ``'<NN>'``, not
          ``'<NN'`` followed by one or more repetitions of ``'>'``.
        - Whitespace in tag patterns is ignored.  So
          ``'<DT> | <NN>'`` is equivalent to ``'<DT>|<NN>'``
        - In tag patterns, ``'.'`` is equivalent to ``'[^{}<>]'``; so
          ``'<NN.*>'`` matches any single tag starting with ``'NN'``.

    In particular, ``tag_pattern2re_pattern`` performs the following
    transformations on the given pattern:

        - Replace '.' with '[^<>{}]'
        - Remove any whitespace
        - Add extra parens around '<' and '>', to make '<' and '>' act
          like parentheses.  E.g., so that in '<NN>+', the '+' has scope
          over the entire '<NN>'; and so that in '<NN|IN>', the '|' has
          scope over 'NN' and 'IN', but not '<' or '>'.
        - Check to make sure the resulting pattern is valid.

    :type tag_pattern: str
    :param tag_pattern: The tag pattern to convert to a regular
        expression pattern.
    :raise ValueError: If ``tag_pattern`` is not a valid tag pattern.
        In particular, ``tag_pattern`` should not include braces; and it
        should not contain nested or mismatched angle-brackets.
    :rtype: str
    :return: A regular expression pattern corresponding to
        ``tag_pattern``.
    """
    tag_pattern = re.sub('\\s', '', tag_pattern)
    tag_pattern = re.sub('<', '(<(', tag_pattern)
    tag_pattern = re.sub('>', ')>)', tag_pattern)
    if not CHUNK_TAG_PATTERN.match(tag_pattern):
        raise ValueError('Bad tag pattern: %r' % tag_pattern)

    def reverse_str(str):
        lst = list(str)
        lst.reverse()
        return ''.join(lst)
    tc_rev = reverse_str(ChunkString.CHUNK_TAG_CHAR)
    reversed = reverse_str(tag_pattern)
    reversed = re.sub('\\.(?!\\\\(\\\\\\\\)*($|[^\\\\]))', tc_rev, reversed)
    tag_pattern = reverse_str(reversed)
    return tag_pattern