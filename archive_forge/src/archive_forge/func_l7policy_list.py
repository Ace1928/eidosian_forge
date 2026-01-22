import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def l7policy_list(self, **kwargs):
    """List all l7policies

        :param kwargs:
            Parameters to filter on
        :return:
            List of l7policies
        """
    url = const.BASE_L7POLICY_URL
    response = self._list(url, get_all=True, resources=const.L7POLICY_RESOURCES, **kwargs)
    return response