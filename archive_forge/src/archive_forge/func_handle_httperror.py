from __future__ import (absolute_import, division, print_function)
from abc import abstractmethod
from ansible.plugins import AnsiblePlugin
def handle_httperror(self, exc):
    """Overridable method for dealing with HTTP codes.

        This method will attempt to handle known cases of HTTP status codes.
        If your API uses status codes to convey information in a regular way,
        you can override this method to handle it appropriately.

        :returns:
            * True if the code has been handled in a way that the request
            may be resent without changes.
            * False if the error cannot be handled or recovered from by the
            plugin. This will result in the HTTPError being raised as an
            exception for the caller to deal with as appropriate (most likely
            by failing).
            * Any other value returned is taken as a valid response from the
            server without making another request. In many cases, this can just
            be the original exception.
            """
    if exc.code == 401:
        if self.connection._auth:
            self.connection._auth = None
            self.login(self.connection.get_option('remote_user'), self.connection.get_option('password'))
            return True
        else:
            return False
    return exc