import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def quota_defaults_show(self):
    """Show quota defaults

        :return:
            A ``dict`` representing a list of quota defaults
        """
    url = const.BASE_QUOTA_DEFAULT_URL
    response = self._list(url)
    return response