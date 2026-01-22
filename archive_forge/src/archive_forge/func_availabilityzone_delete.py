import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def availabilityzone_delete(self, availabilityzone_name):
    """Delete a availabilityzone

        :param string availabilityzone_name:
            Name of the availabilityzone to delete
        :return:
            Response Code from the API
        """
    url = const.BASE_SINGLE_AVAILABILITYZONE_URL.format(name=availabilityzone_name)
    response = self._delete(url)
    return response