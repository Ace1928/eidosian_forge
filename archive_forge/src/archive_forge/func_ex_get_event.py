from libcloud.utils.py3 import httplib, parse_qs, urlparse
from libcloud.common.base import BaseDriver, JsonResponse, ConnectionKey
from libcloud.common.types import LibcloudError, InvalidCredsError
def ex_get_event(self, event_id):
    """
        Get an event object

        :param      event_id: Event id (required)
        :type       event_id: ``str``
        """
    params = {}
    return self.connection.request('/v2/actions/%s' % event_id, params=params).object['action']