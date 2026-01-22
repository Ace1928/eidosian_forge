import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def load_balancer_stats_show(self, lb_id, **kwargs):
    """Shows the current statistics for a load balancer.

        :param string lb_id:
            ID of the load balancer
        :return:
            A dict of the specified load balancer's statistics
        """
    url = const.BASE_LB_STATS_URL.format(uuid=lb_id)
    response = self._list(url, **kwargs)
    return response