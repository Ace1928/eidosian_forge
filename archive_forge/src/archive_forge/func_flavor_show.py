import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def flavor_show(self, flavor_id):
    """Show a flavor

        :param string flavor_id:
            ID of the flavor to show
        :return:
            A dict of the specified flavor's settings
        """
    response = self._find(path=const.BASE_FLAVOR_URL, value=flavor_id)
    return response