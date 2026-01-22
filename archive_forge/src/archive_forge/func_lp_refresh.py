from email.message import Message
from io import BytesIO
from json import dumps, loads
import sys
from wadllib.application import Resource as WadlResource
from lazr.restfulclient import __version__
from lazr.restfulclient._browser import Browser, RestfulHttp
from lazr.restfulclient._json import DatetimeJSONEncoder
from lazr.restfulclient.errors import HTTPError
from lazr.uri import URI
def lp_refresh(self, new_url=None):
    """Update this resource's representation."""
    etag = getattr(self, 'http_etag', None)
    super(Entry, self).lp_refresh(new_url, etag)
    self._dirty_attributes.clear()