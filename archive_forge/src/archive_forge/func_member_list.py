import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def member_list(self, pool_id, **kwargs):
    """Lists the member from a given pool id

        :param pool_id:
            ID of the pool
        :param kwargs:
            A dict of filter arguments
        :return:
            Response list members
        """
    url = const.BASE_MEMBER_URL.format(pool_id=pool_id)
    response = self._list(url, get_all=True, resources=const.MEMBER_RESOURCES, **kwargs)
    return response