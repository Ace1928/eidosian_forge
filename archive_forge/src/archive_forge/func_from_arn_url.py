import collections
import re
from oslo_utils import encodeutils
from urllib import parse as urlparse
from heat.common.i18n import _
@classmethod
def from_arn_url(cls, url):
    """Generate a new HeatIdentifier by parsing the supplied URL.

        The URL is expected to contain a valid arn as part of the path.
        """
    urlp = urlparse.urlparse(url)
    if urlp.scheme not in ('http', 'https') or not urlp.netloc or (not urlp.path):
        raise ValueError(_('"%s" is not a valid URL') % url)
    arn_url_prefix = '/arn%3Aopenstack%3Aheat%3A%3A'
    match = re.search(arn_url_prefix, urlp.path, re.IGNORECASE)
    if match is None:
        raise ValueError(_('"%s" is not a valid ARN URL') % url)
    url_arn = urlp.path[match.start() + 1:]
    arn = urlparse.unquote(url_arn)
    return cls.from_arn(arn)