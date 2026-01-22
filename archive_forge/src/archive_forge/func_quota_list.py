import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def quota_list(self, **params):
    """List all quotas

        :param params:
            Parameters to filter on (not implemented)
        :return:
            A ``dict`` representing a list of quotas for the project
        """
    url = const.BASE_QUOTA_URL
    response = self._list(url, get_all=True, resources=const.QUOTA_RESOURCES, **params)
    return response