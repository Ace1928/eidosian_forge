import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def provider_list(self):
    """List all providers

        :return:
            A ``dict`` containing a list of provider
        """
    url = const.BASE_PROVIDER_URL
    response = self._list(path=url, get_all=True, resources=const.PROVIDER_RESOURCES)
    return response