import logging
import threading
import enum
from oslo_utils import reflection
from glance_store import exceptions
from glance_store.i18n import _LW
def op_checker(store, *args, **kwargs):
    get_capabilities = [BitMasks.READ_ACCESS, BitMasks.READ_OFFSET if kwargs.get('offset') else BitMasks.NONE, BitMasks.READ_CHUNK if kwargs.get('chunk_size') else BitMasks.NONE]
    op_cap_map = {'get': get_capabilities, 'add': [BitMasks.WRITE_ACCESS], 'delete': [BitMasks.WRITE_ACCESS]}
    op_exec_map = {'get': exceptions.StoreRandomGetNotSupported if kwargs.get('offset') or kwargs.get('chunk_size') else exceptions.StoreGetNotSupported, 'add': exceptions.StoreAddDisabled, 'delete': exceptions.StoreDeleteNotSupported}
    op = store_op_fun.__name__.lower()
    try:
        req_cap = op_cap_map[op]
    except KeyError:
        LOG.warning(_LW('The capability of operation "%s" could not be checked.'), op)
    else:
        if not store.is_capable(*req_cap):
            kwargs.setdefault('offset', 0)
            kwargs.setdefault('chunk_size', None)
            raise op_exec_map[op](**kwargs)
    return store_op_fun(store, *args, **kwargs)