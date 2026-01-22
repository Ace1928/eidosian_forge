import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def availabilityzoneprofile_show(self, availabilityzoneprofile_id):
    """Show a availabilityzone profile

        :param string availabilityzoneprofile_id:
            ID of the availabilityzone profile to show
        :return:
            A dict of the specified availabilityzone profile's settings
        """
    response = self._find(path=const.BASE_AVAILABILITYZONEPROFILE_URL, value=availabilityzoneprofile_id)
    return response