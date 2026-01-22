import json
import zaqarclient.transport.errors as errors
def claim_delete(transport, request, queue_name, claim_id):
    """Deletes a Claim `claim_id`

    :param transport: Transport instance to use
    :type transport: `transport.base.Transport`
    :param request: Request instance ready to be sent.
    :type request: `transport.request.Request`
    """
    request.operation = 'claim_delete'
    request.params['queue_name'] = queue_name
    request.params['claim_id'] = claim_id
    transport.send(request)