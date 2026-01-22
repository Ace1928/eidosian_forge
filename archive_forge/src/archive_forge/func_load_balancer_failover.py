import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def load_balancer_failover(self, lb_id):
    """Trigger load balancer failover

        :param string lb_id:
            ID of the load balancer to failover
        :return:
            Response Code from the API
        """
    url = const.BASE_LOADBALANCER_FAILOVER_URL.format(uuid=lb_id)
    response = self._create(url, method='PUT')
    return response