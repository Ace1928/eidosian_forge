from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import codecs
from antlr3.constants import DEFAULT_CHANNEL, EOF
from antlr3.tokens import Token, EOF_TOKEN
import six
from six import StringIO
def setTokenSource(self, tokenSource):
    """Reset this token stream by setting its token source."""
    self.tokenSource = tokenSource
    self.tokens = []
    self.p = -1
    self.channel = DEFAULT_CHANNEL