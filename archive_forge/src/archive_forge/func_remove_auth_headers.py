import itertools
from oslo_serialization import jsonutils
import webob
def remove_auth_headers(self):
    """Remove headers so a user can't fake authentication."""
    for header in self._all_auth_headers():
        self.headers.pop(header, None)