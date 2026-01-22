import re
from . import urlutils
from .hooks import Hooks
def parse_rcp_location(location):
    """Convert a rcp-style location to a URL.

    :param location: Location to convert, e.g. "foo:bar"
    :param scheme: URL scheme to return, defaults to "ssh"
    :return: A URL, e.g. "ssh://foo/bar"
    :raises ValueError: if this is not a RCP-style URL
    """
    m = re.match('^(?P<user>[^@:/]+@)?(?P<host>[^/:]{2,}):(?P<path>.*)$', location)
    if not m:
        raise ValueError('Not a RCP URL')
    if m.group('path').startswith('//'):
        raise ValueError('Not a RCP URL: already looks like a URL')
    return (m.group('host'), m.group('user')[:-1] if m.group('user') else None, m.group('path'))