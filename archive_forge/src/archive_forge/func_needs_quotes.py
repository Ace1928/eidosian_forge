from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import time
import six
@classmethod
def needs_quotes(cls, text):
    """Returns True if text "needs" quotes for pretty printing contents."""
    pairs = {'"': '"', "'": "'", '{': '}', '[': ']', '(': ')'}
    if text == '':
        return True
    if ' ' not in text:
        return False
    return pairs.get(text[0]) != text[-1]