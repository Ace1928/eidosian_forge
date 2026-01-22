from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import codecs
from antlr3.constants import DEFAULT_CHANNEL, EOF
from antlr3.tokens import Token, EOF_TOKEN
import six
from six import StringIO
def skipOffTokenChannels(self, i):
    """
        Given a starting index, return the index of the first on-channel
        token.
        """
    try:
        while self.tokens[i].channel != self.channel:
            i += 1
    except IndexError:
        pass
    return i