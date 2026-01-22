from oslo_utils import uuidutils
from zaqarclient.common import decorators
from zaqarclient.queues.v1 import client
from zaqarclient.queues.v1 import iterator
from zaqarclient.queues.v2 import core
from zaqarclient.queues.v2 import queues
from zaqarclient.queues.v2 import subscription
def queues(self, **params):
    """Gets a list of queues from the server

        :returns: A list of queues
        :rtype: `list`
        """
    req, trans = self._request_and_transport()
    queue_list = core.queue_list(trans, req, **params)
    count = None
    if params.get('with_count'):
        count = queue_list.get('count', None)
    list_iter = iterator._Iterator(self, queue_list, 'queues', self.queues_module.create_object(self))
    return (list_iter, count)