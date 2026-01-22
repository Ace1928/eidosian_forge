import json
import zaqarclient.transport.errors as errors
def message_list(transport, request, queue_name, callback=None, **kwargs):
    """Gets a list of messages in queue `queue_name`

    :param transport: Transport instance to use
    :type transport: `transport.base.Transport`
    :param request: Request instance ready to be sent.
    :type request: `transport.request.Request`
    :param queue_name: Queue reference name.
    :type queue_name: str
    :param callback: Optional callable to use as callback.
        If specified, this request will be sent asynchronously.
        (IGNORED UNTIL ASYNC SUPPORT IS COMPLETE)
    :type callback: Callable object.
    :param kwargs: Optional arguments for this operation.
        - marker: Where to start getting messages from.
        - limit: Maximum number of messages to get.
        - echo: Whether to get our own messages.
        - include_claimed: Whether to include claimed
            messages.
    """
    request.operation = 'message_list'
    request.params['queue_name'] = queue_name
    request.params.update(kwargs)
    resp = transport.send(request)
    if not resp.content:
        return {'links': [], 'messages': []}
    return resp.deserialized_content