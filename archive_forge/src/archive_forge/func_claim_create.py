import json
import zaqarclient.transport.errors as errors
def claim_create(transport, request, queue_name, **kwargs):
    """Creates a Claim `claim_id` on the queue `queue_name`

    :param transport: Transport instance to use
    :type transport: `transport.base.Transport`
    :param request: Request instance ready to be sent.
    :type request: `transport.request.Request`
    """
    request.operation = 'claim_create'
    request.params['queue_name'] = queue_name
    if 'limit' in kwargs:
        request.params['limit'] = kwargs.pop('limit')
    request.content = json.dumps(kwargs)
    resp = transport.send(request)
    return resp.deserialized_content