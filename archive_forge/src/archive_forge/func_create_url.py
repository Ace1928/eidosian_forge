import requests
from ._compat import urljoin
def create_url(self, url):
    """Create the URL based off this partial path."""
    return urljoin(self.base_url, url)