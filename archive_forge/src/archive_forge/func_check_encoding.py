import sys
import os
import re
import codecs
from docutils import TransformSpec
from docutils.utils.error_reporting import locale_encoding, ErrorString, ErrorOutput
def check_encoding(stream, encoding):
    """Test, whether the encoding of `stream` matches `encoding`.

    Returns

    :None:  if `encoding` or `stream.encoding` are not a valid encoding
            argument (e.g. ``None``) or `stream.encoding is missing.
    :True:  if the encoding argument resolves to the same value as `encoding`,
    :False: if the encodings differ.
    """
    try:
        return codecs.lookup(stream.encoding) == codecs.lookup(encoding)
    except (LookupError, AttributeError, TypeError):
        return None