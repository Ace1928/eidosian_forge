import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def health_monitor_list(self, **kwargs):
    """List all health monitors

        :param kwargs:
            Parameters to filter on
        :return:
            A dict containing a list of health monitors
        """
    url = const.BASE_HEALTH_MONITOR_URL
    response = self._list(url, get_all=True, resources=const.HEALTH_MONITOR_RESOURCES, **kwargs)
    return response