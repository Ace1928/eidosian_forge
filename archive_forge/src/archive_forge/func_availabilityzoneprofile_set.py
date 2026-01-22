import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def availabilityzoneprofile_set(self, availabilityzoneprofile_id, **kwargs):
    """Update a availabilityzone profile's settings

        :param string availabilityzoneprofile_id:
            ID of the availabilityzone profile to update
        :kwargs:
            A dict of arguments to update the availabilityzone profile
        :return:
            Response Code from the API
        """
    url = const.BASE_SINGLE_AVAILABILITYZONEPROFILE_URL.format(uuid=availabilityzoneprofile_id)
    response = self._create(url, method='PUT', **kwargs)
    return response