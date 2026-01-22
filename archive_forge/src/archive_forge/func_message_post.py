import json
import zaqarclient.transport.errors as errors
def message_post(transport, request, queue_name, messages, callback=None):
    """Post messages to `queue_name`

    :param transport: Transport instance to use
    :type transport: `transport.base.Transport`
    :param request: Request instance ready to be sent.
    :type request: `transport.request.Request`
    :param queue_name: Queue reference name.
    :type queue_name: str
    :param messages: One or more messages to post.
    :param messages: `list`
    :param callback: Optional callable to use as callback.
        If specified, this request will be sent asynchronously.
        (IGNORED UNTIL ASYNC SUPPORT IS COMPLETE)
    :type callback: Callable object.
    """
    request.operation = 'message_post'
    request.params['queue_name'] = queue_name
    request.content = json.dumps(messages)
    resp = transport.send(request)
    return resp.deserialized_content