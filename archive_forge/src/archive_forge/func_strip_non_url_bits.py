from __future__ import unicode_literals
import re
from tensorboard._vendor import html5lib
from tensorboard._vendor.html5lib.filters.base import Filter
from tensorboard._vendor.html5lib.filters.sanitizer import allowed_protocols
from tensorboard._vendor.html5lib.serializer import HTMLSerializer
from tensorboard._vendor.bleach import callbacks as linkify_callbacks
from tensorboard._vendor.bleach.encoding import force_unicode
from tensorboard._vendor.bleach.utils import alphabetize_attributes
def strip_non_url_bits(self, fragment):
    """Strips non-url bits from the url

        This accounts for over-eager matching by the regex.

        """
    prefix = suffix = ''
    while fragment:
        if fragment.startswith(u'('):
            prefix = prefix + u'('
            fragment = fragment[1:]
            if fragment.endswith(u')'):
                suffix = u')' + suffix
                fragment = fragment[:-1]
            continue
        if fragment.endswith(u')') and u'(' not in fragment:
            fragment = fragment[:-1]
            suffix = u')' + suffix
            continue
        if fragment.endswith(u','):
            fragment = fragment[:-1]
            suffix = u',' + suffix
            continue
        if fragment.endswith(u'.'):
            fragment = fragment[:-1]
            suffix = u'.' + suffix
            continue
        break
    return (fragment, prefix, suffix)