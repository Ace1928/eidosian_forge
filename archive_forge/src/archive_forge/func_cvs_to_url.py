import re
from . import urlutils
from .hooks import Hooks
def cvs_to_url(location):
    """Convert a CVS pserver location string to a URL.

    :param location: pserver URL
    :return: A cvs+pserver URL
    """
    try:
        scheme, host, user, path = parse_cvs_location(location)
    except ValueError as e:
        raise urlutils.InvalidURL(path=location, extra=str(e))
    return str(urlutils.URL(scheme='cvs+' + scheme, quoted_user=urlutils.quote(user) if user else None, quoted_host=urlutils.quote(host), quoted_password=None, port=None, quoted_path=urlutils.quote(path)))