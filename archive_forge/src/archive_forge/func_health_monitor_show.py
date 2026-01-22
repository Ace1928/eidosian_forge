import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def health_monitor_show(self, health_monitor_id):
    """Show a health monitor's settings

        :param string health_monitor_id:
            ID of the health monitor to show
        :return:
            Dict of the specified health monitor's settings
        """
    url = const.BASE_HEALTH_MONITOR_URL
    response = self._find(path=url, value=health_monitor_id)
    return response