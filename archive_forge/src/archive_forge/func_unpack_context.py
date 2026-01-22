import collections
import uuid
from oslo_config import cfg
from oslo_messaging._drivers import common as rpc_common
def unpack_context(msg):
    """Unpack context from msg."""
    context_dict = {}
    for key in list(msg.keys()):
        key = str(key)
        if key.startswith('_context_'):
            value = msg.pop(key)
            context_dict[key[9:]] = value
    context_dict['msg_id'] = msg.pop('_msg_id', None)
    context_dict['reply_q'] = msg.pop('_reply_q', None)
    context_dict['client_timeout'] = msg.pop('_timeout', None)
    return RpcContext.from_dict(context_dict)