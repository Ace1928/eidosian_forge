import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def flavorprofile_delete(self, flavorprofile_id):
    """Delete a flavor profile

        :param string flavorprofile_id:
            ID of the flavor profile to delete
        :return:
            Response Code from the API
        """
    url = const.BASE_SINGLE_FLAVORPROFILE_URL.format(uuid=flavorprofile_id)
    response = self._delete(url)
    return response