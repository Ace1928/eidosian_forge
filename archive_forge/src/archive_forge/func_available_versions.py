import warnings
from debtcollector import removals
from keystoneauth1 import plugin
from keystoneclient import _discover
from keystoneclient import exceptions
from keystoneclient.i18n import _
from keystoneclient import session as client_session
from keystoneclient.v2_0 import client as v2_client
from keystoneclient.v3 import client as v3_client
@removals.remove(message='Use raw_version_data instead.', version='1.7.0', removal_version='2.0.0')
def available_versions(self, **kwargs):
    """Return a list of identity APIs available on the server.

        The list returned includes the data associated with them.

        .. warning::

            This method is deprecated as of the 1.7.0 release in favor of
            :meth:`raw_version_data` and may be removed in the 2.0.0 release.

        :param bool unstable: Accept endpoints not marked 'stable'. (optional)
                              Equates to setting allow_experimental
                              and allow_unknown to True.
        :param bool allow_experimental: Allow experimental version endpoints.
        :param bool allow_deprecated: Allow deprecated version endpoints.
        :param bool allow_unknown: Allow endpoints with an unrecognised status.

        :returns: A List of dictionaries as presented by the server. Each dict
                  will contain the version and the URL to use for the version.
                  It is a direct representation of the layout presented by the
                  identity API.
        """
    return self.raw_version_data(**kwargs)