import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def flavor_set(self, flavor_id, **kwargs):
    """Update a flavor's settings

        :param string flavor_id:
            ID of the flavor to update
        :param kwargs:
            A dict of arguments to update a flavor
        :return:
            Response Code from the API
        """
    url = const.BASE_SINGLE_FLAVOR_URL.format(uuid=flavor_id)
    response = self._create(url, method='PUT', **kwargs)
    return response