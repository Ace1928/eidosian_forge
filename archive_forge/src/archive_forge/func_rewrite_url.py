from .. import osutils, tests, urlutils
from ..directory_service import directories
from ..location import hooks as location_hooks
from ..location import location_to_url, rcp_location_to_url
def rewrite_url(url, purpose=None):
    return url.replace('foo', 'bar')