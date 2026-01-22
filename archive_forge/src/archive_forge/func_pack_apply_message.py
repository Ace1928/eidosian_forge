import pickle
import warnings
from itertools import chain
from jupyter_client.session import MAX_BYTES, MAX_ITEMS
def pack_apply_message(f, args, kwargs, buffer_threshold=MAX_BYTES, item_threshold=MAX_ITEMS):
    """pack up a function, args, and kwargs to be sent over the wire

    Each element of args/kwargs will be canned for special treatment,
    but inspection will not go any deeper than that.

    Any object whose data is larger than `threshold`  will not have their data copied
    (only numpy arrays and bytes/buffers support zero-copy)

    Message will be a list of bytes/buffers of the format:

    [ cf, pinfo, <arg_bufs>, <kwarg_bufs> ]

    With length at least two + len(args) + len(kwargs)
    """
    arg_bufs = list(chain.from_iterable((serialize_object(arg, buffer_threshold, item_threshold) for arg in args)))
    kw_keys = sorted(kwargs.keys())
    kwarg_bufs = list(chain.from_iterable((serialize_object(kwargs[key], buffer_threshold, item_threshold) for key in kw_keys)))
    info = dict(nargs=len(args), narg_bufs=len(arg_bufs), kw_keys=kw_keys)
    msg = [pickle.dumps(can(f), PICKLE_PROTOCOL)]
    msg.append(pickle.dumps(info, PICKLE_PROTOCOL))
    msg.extend(arg_bufs)
    msg.extend(kwarg_bufs)
    return msg