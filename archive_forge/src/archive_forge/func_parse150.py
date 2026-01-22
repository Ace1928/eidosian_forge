import sys
import socket
from socket import _GLOBAL_DEFAULT_TIMEOUT
def parse150(resp):
    """Parse the '150' response for a RETR request.
    Returns the expected transfer size or None; size is not guaranteed to
    be present in the 150 message.
    """
    if resp[:3] != '150':
        raise error_reply(resp)
    global _150_re
    if _150_re is None:
        import re
        _150_re = re.compile('150 .* \\((\\d+) bytes\\)', re.IGNORECASE | re.ASCII)
    m = _150_re.match(resp)
    if not m:
        return None
    return int(m.group(1))