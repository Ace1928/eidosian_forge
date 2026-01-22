import datetime
import json
from oslo_utils import timeutils
from zaqarclient.queues.v1 import core
def queue_purge(transport, request, name, resource_types=None):
    """Purge resources under a queue

    :param transport: Transport instance to use
    :type transport: `transport.base.Transport`
    :param request: Request instance ready to be sent.
    :type request: `transport.request.Request`
    :param name: Queue reference name.
    :type name: str
    :param resource_types: Resource types will be purged
    :type resource_types: `list`
    """
    request.operation = 'queue_purge'
    request.params['queue_name'] = name
    if resource_types:
        request.content = json.dumps({'resource_types': resource_types})
    resp = transport.send(request)
    return resp.deserialized_content