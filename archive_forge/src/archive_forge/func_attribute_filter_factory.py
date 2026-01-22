from __future__ import unicode_literals
import re
from xml.sax.saxutils import unescape
from tensorboard._vendor import html5lib
from tensorboard._vendor.html5lib.constants import namespaces
from tensorboard._vendor.html5lib.filters import sanitizer
from tensorboard._vendor.html5lib.serializer import HTMLSerializer
from tensorboard._vendor.bleach.encoding import force_unicode
from tensorboard._vendor.bleach.utils import alphabetize_attributes
def attribute_filter_factory(attributes):
    """Generates attribute filter function for the given attributes value

    The attributes value can take one of several shapes. This returns a filter
    function appropriate to the attributes value. One nice thing about this is
    that there's less if/then shenanigans in the ``allow_token`` method.

    """
    if callable(attributes):
        return attributes
    if isinstance(attributes, dict):

        def _attr_filter(tag, attr, value):
            if tag in attributes:
                attr_val = attributes[tag]
                if callable(attr_val):
                    return attr_val(tag, attr, value)
                if attr in attr_val:
                    return True
            if '*' in attributes:
                attr_val = attributes['*']
                if callable(attr_val):
                    return attr_val(tag, attr, value)
                return attr in attr_val
            return False
        return _attr_filter
    if isinstance(attributes, list):

        def _attr_filter(tag, attr, value):
            return attr in attributes
        return _attr_filter
    raise ValueError('attributes needs to be a callable, a list or a dict')