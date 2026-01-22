import itertools
from oslo_serialization import jsonutils
import webob
@property
def auth_type(self):
    """The authentication type that was performed by the web server.

        The returned string value is always lower case.

        :returns: The AUTH_TYPE environ string or None if not present.
        :rtype: str or None
        """
    try:
        auth_type = self.environ['AUTH_TYPE']
    except KeyError:
        return None
    else:
        return auth_type.lower()