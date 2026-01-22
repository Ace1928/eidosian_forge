import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def header_id_from_text(self, text, prefix, n):
    """Generate a header id attribute value from the given header
        HTML content.

        This is only called if the "header-ids" extra is enabled.
        Subclasses may override this for different header ids.

        @param text {str} The text of the header tag
        @param prefix {str} The requested prefix for header ids. This is the
            value of the "header-ids" extra key, if any. Otherwise, None.
        @param n {int} The <hN> tag number, i.e. `1` for an <h1> tag.
        @returns {str} The value for the header tag's "id" attribute. Return
            None to not have an id attribute and to exclude this header from
            the TOC (if the "toc" extra is specified).
        """
    header_id = _slugify(text)
    if prefix and isinstance(prefix, str):
        header_id = prefix + '-' + header_id
    self._count_from_header_id[header_id] += 1
    if 0 == len(header_id) or self._count_from_header_id[header_id] > 1:
        header_id += '-%s' % self._count_from_header_id[header_id]
    return header_id