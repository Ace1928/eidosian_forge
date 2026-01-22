import builtins
import copy
import os
import pickle
import contextlib
import subprocess
import socket
import parlai.utils.logging as logging
def sync_object(data, max_size=16384):
    """
    Sync an object among all workers.

    All workers will return the same value for `data` when returning from this
    method, always using the primary worker's version. Useful for ensuring control
    flow decisions are made the same.

    :param object data:
        The object to synchronize. Must be pickleable.
    :param int max_size:
        The maximum size of this object in bytes. Large values than 255^2 are not
        supported.

    :return: the synchronized data
    """
    if not is_distributed():
        return data
    if not hasattr(sync_object, '_buffer') or sync_object._buffer.numel() < max_size:
        sync_object._buffer = torch.cuda.ByteTensor(max_size)
    buffer = sync_object._buffer
    if is_primary_worker():
        enc = pickle.dumps(data)
        enc_size = len(enc)
        if enc_size + 2 > max_size or enc_size > 255 * 255:
            raise ValueError('encoded data exceeds max_size')
        buffer[0] = enc_size // 255
        buffer[1] = enc_size % 255
        buffer[2:enc_size + 2] = torch.ByteTensor(list(enc))
    dist.broadcast(buffer, 0)
    if not is_primary_worker():
        enc_size = buffer[0].item() * 255 + buffer[1].item()
        try:
            data = pickle.loads(bytes(buffer[2:enc_size + 2].tolist()))
        except pickle.UnpicklingError:
            raise RuntimeError('There was an unpickling error in sync_object. This likely means your workers got out of syncronization (e.g. one is expecting to sync and another is not.)')
    return data