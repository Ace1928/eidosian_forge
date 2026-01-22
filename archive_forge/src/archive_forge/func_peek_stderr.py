from kubernetes.client.rest import ApiException
import select
import certifi
import time
import collections
from websocket import WebSocket, ABNF, enableTrace
import six
import ssl
from six.moves.urllib.parse import urlencode, quote_plus, urlparse, urlunparse
def peek_stderr(self, timeout=0):
    """Same as peek_channel with channel=2."""
    return self.peek_channel(STDERR_CHANNEL, timeout=timeout)