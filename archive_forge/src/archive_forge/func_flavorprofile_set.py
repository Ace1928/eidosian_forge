import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def flavorprofile_set(self, flavorprofile_id, **kwargs):
    """Update a flavor profile's settings

        :param string flavorprofile_id:
            ID of the flavor profile to update
        :kwargs:
            A dict of arguments to update the flavor profile
        :return:
            Response Code from the API
        """
    url = const.BASE_SINGLE_FLAVORPROFILE_URL.format(uuid=flavorprofile_id)
    response = self._create(url, method='PUT', **kwargs)
    return response