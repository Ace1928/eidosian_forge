from kubernetes.client.rest import ApiException
import select
import certifi
import time
import collections
from websocket import WebSocket, ABNF, enableTrace
import six
import ssl
from six.moves.urllib.parse import urlencode, quote_plus, urlparse, urlunparse
def peek_stdout(self, timeout=0):
    """Same as peek_channel with channel=1."""
    return self.peek_channel(STDOUT_CHANNEL, timeout=timeout)