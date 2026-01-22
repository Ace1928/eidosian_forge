import bisect
import os
import pickle
import re
import tempfile
from functools import reduce
from xml.etree import ElementTree
from nltk.data import (
from nltk.internals import slice_bounds
from nltk.tokenize import wordpunct_tokenize
from nltk.util import AbstractLazySequence, LazyConcatenation, LazySubsequence
def read_sexpr_block(stream, block_size=16384, comment_char=None):
    """
    Read a sequence of s-expressions from the stream, and leave the
    stream's file position at the end the last complete s-expression
    read.  This function will always return at least one s-expression,
    unless there are no more s-expressions in the file.

    If the file ends in in the middle of an s-expression, then that
    incomplete s-expression is returned when the end of the file is
    reached.

    :param block_size: The default block size for reading.  If an
        s-expression is longer than one block, then more than one
        block will be read.
    :param comment_char: A character that marks comments.  Any lines
        that begin with this character will be stripped out.
        (If spaces or tabs precede the comment character, then the
        line will not be stripped.)
    """
    start = stream.tell()
    block = stream.read(block_size)
    encoding = getattr(stream, 'encoding', None)
    assert encoding is not None or isinstance(block, str)
    if encoding not in (None, 'utf-8'):
        import warnings
        warnings.warn('Parsing may fail, depending on the properties of the %s encoding!' % encoding)
    if comment_char:
        COMMENT = re.compile('(?m)^%s.*$' % re.escape(comment_char))
    while True:
        try:
            if comment_char:
                block += stream.readline()
                block = re.sub(COMMENT, _sub_space, block)
            tokens, offset = _parse_sexpr_block(block)
            offset = re.compile('\\s*').search(block, offset).end()
            if encoding is None:
                stream.seek(start + offset)
            else:
                stream.seek(start + len(block[:offset].encode(encoding)))
            return tokens
        except ValueError as e:
            if e.args[0] == 'Block too small':
                next_block = stream.read(block_size)
                if next_block:
                    block += next_block
                    continue
                else:
                    return [block.strip()]
            else:
                raise