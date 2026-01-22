import json
import zaqarclient.transport.errors as errors
def claim_update(transport, request, queue_name, claim_id, **kwargs):
    """Updates a Claim `claim_id`

    :param transport: Transport instance to use
    :type transport: `transport.base.Transport`
    :param request: Request instance ready to be sent.
    :type request: `transport.request.Request`
    """
    request.operation = 'claim_update'
    request.params['queue_name'] = queue_name
    request.params['claim_id'] = claim_id
    request.content = json.dumps(kwargs)
    resp = transport.send(request)
    return resp.deserialized_content