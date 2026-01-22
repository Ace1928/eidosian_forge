import datetime
import json
from oslo_utils import timeutils
from zaqarclient.queues.v1 import core
def queue_update(transport, request, name, metadata, callback=None):
    """Updates a queue's metadata using PATCH for API v2

    :param transport: Transport instance to use
    :type transport: `transport.base.Transport`
    :param request: Request instance ready to be sent.
    :type request: `transport.request.Request`
    :param name: Queue reference name.
    :type name: str
    :param metadata: Queue's metadata object.
    :type metadata: `list`
    :param callback: Optional callable to use as callback.
        If specified, this request will be sent asynchronously.
        (IGNORED UNTIL ASYNC SUPPORT IS COMPLETE)
    :type callback: Callable object.
    """
    request.operation = 'queue_update'
    request.params['queue_name'] = name
    request.content = json.dumps(metadata)
    resp = transport.send(request)
    return resp.deserialized_content