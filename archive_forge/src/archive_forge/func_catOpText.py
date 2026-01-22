from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import codecs
from antlr3.constants import DEFAULT_CHANNEL, EOF
from antlr3.tokens import Token, EOF_TOKEN
import six
from six import StringIO
def catOpText(self, a, b):
    x = ''
    y = ''
    if a is not None:
        x = a
    if b is not None:
        y = b
    return x + y