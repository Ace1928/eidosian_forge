import io
import urllib3
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import logger
from sentry_sdk.envelope import Envelope
def setup_spotlight(options):
    url = options.get('spotlight')
    if isinstance(url, str):
        pass
    elif url is True:
        url = 'http://localhost:8969/stream'
    else:
        return None
    return SpotlightClient(url)