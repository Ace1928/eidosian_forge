from collections import defaultdict
import itertools
import re
import warnings
import sys
from bs4.element import (
from . import _htmlparser
def set_up_substitutions(self, tag):
    """Replace the declared encoding in a <meta> tag with a placeholder,
        to be substituted when the tag is output to a string.

        An HTML document may come in to Beautiful Soup as one
        encoding, but exit in a different encoding, and the <meta> tag
        needs to be changed to reflect this.

        :param tag: A `Tag`
        :return: Whether or not a substitution was performed.
        """
    if tag.name != 'meta':
        return False
    http_equiv = tag.get('http-equiv')
    content = tag.get('content')
    charset = tag.get('charset')
    meta_encoding = None
    if charset is not None:
        meta_encoding = charset
        tag['charset'] = CharsetMetaAttributeValue(charset)
    elif content is not None and http_equiv is not None and (http_equiv.lower() == 'content-type'):
        tag['content'] = ContentMetaAttributeValue(content)
    return meta_encoding is not None