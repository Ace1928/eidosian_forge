import re
import sys
import urllib   # For urllib.parse.unquote
from string import hexdigits
from operator import itemgetter
from email import _encoded_words as _ew
from email import errors
from email import utils
@property
def stripped_value(self):
    token = self[0]
    if token.token_type == 'cfws':
        token = self[1]
    if token.token_type.endswith(('quoted-string', 'attribute', 'extended-attribute')):
        return token.stripped_value
    return self.value