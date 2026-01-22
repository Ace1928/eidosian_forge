import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def flavorprofile_show(self, flavorprofile_id):
    """Show a flavor profile

        :param string flavorprofile_id:
            ID of the flavor profile to show
        :return:
            A dict of the specified flavor profile's settings
        """
    response = self._find(path=const.BASE_FLAVORPROFILE_URL, value=flavorprofile_id)
    return response