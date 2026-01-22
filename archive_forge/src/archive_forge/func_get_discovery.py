import abc
import base64
import functools
import hashlib
import json
import threading
from keystoneauth1 import _utils as utils
from keystoneauth1 import access
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import plugin
def get_discovery(self, session, url, authenticated=None):
    """Return the discovery object for a URL.

        Check the session and the plugin cache to see if we have already
        performed discovery on the URL and if so return it, otherwise create
        a new discovery object, cache it and return it.

        This function is expected to be used by subclasses and should not
        be needed by users.

        :param session: A session object to discover with.
        :type session: keystoneauth1.session.Session
        :param str url: The url to lookup.
        :param bool authenticated: Include a token in the discovery call.
                                   (optional) Defaults to None (use a token
                                   if a plugin is installed).

        :raises keystoneauth1.exceptions.discovery.DiscoveryFailure:
            if for some reason the lookup fails.
        :raises keystoneauth1.exceptions.http.HttpError: An error from an
                                                         invalid HTTP response.

        :returns: A discovery object with the results of looking up that URL.
        """
    return discover.get_discovery(session=session, url=url, cache=self._discovery_cache, authenticated=authenticated)