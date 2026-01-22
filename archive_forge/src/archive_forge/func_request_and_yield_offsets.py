import bisect
import errno
import itertools
import os
import random
import stat
import sys
import time
import warnings
from .. import config, debug, errors, urlutils
from ..errors import LockError, ParamikoNotPresent, PathError, TransportError
from ..osutils import fancy_rename
from ..trace import mutter, warning
from ..transport import (ConnectedTransport, FileExists, FileFileStream,
def request_and_yield_offsets(self, fp):
    """Request the data from the remote machine, yielding the results.

        :param fp: A Paramiko SFTPFile object that supports readv.
        :return: Yield the data requested by the original readv caller, one by
            one.
        """
    requests = self._get_requests()
    offset_iter = iter(self.original_offsets)
    cur_offset, cur_size = next(offset_iter)
    input_start = None
    last_end = None
    buffered_data = []
    buffered_len = 0
    data_chunks = []
    data_stream = itertools.chain(fp.readv(requests), itertools.repeat(None))
    for (start, length), data in zip(requests, data_stream):
        if data is None:
            if cur_coalesced is not None:
                raise errors.ShortReadvError(self.relpath, start, length, len(data))
        if len(data) != length:
            raise errors.ShortReadvError(self.relpath, start, length, len(data))
        self._report_activity(length, 'read')
        if last_end is None:
            buffered_data = [data]
            buffered_len = length
            input_start = start
        elif start == last_end:
            buffered_data.append(data)
            buffered_len += length
        else:
            if buffered_len > 0:
                buffered = b''.join(buffered_data)
                data_chunks.append((input_start, buffered))
            input_start = start
            buffered_data = [data]
            buffered_len = length
        last_end = start + length
        if input_start == cur_offset and cur_size <= buffered_len:
            buffered = b''.join(buffered_data)
            del buffered_data[:]
            buffered_offset = 0
            while input_start == cur_offset and buffered_offset + cur_size <= buffered_len:
                cur_data = buffered[buffered_offset:buffered_offset + cur_size]
                buffered_offset += cur_size
                input_start += cur_size
                yield (cur_offset, cur_data)
                try:
                    cur_offset, cur_size = next(offset_iter)
                except StopIteration:
                    return
            if buffered_offset == len(buffered_data):
                buffered_data = []
                buffered_len = 0
            else:
                buffered = buffered[buffered_offset:]
                buffered_data = [buffered]
                buffered_len = len(buffered)
    fp.close()
    if buffered_len:
        buffered = b''.join(buffered_data)
        del buffered_data[:]
        data_chunks.append((input_start, buffered))
    if data_chunks:
        if 'sftp' in debug.debug_flags:
            mutter('SFTP readv left with %d out-of-order bytes', sum((len(x[1]) for x in data_chunks)))
        while True:
            idx = bisect.bisect_left(data_chunks, (cur_offset,))
            if idx < len(data_chunks) and data_chunks[idx][0] == cur_offset:
                data = data_chunks[idx][1][:cur_size]
            elif idx > 0:
                idx -= 1
                sub_offset = cur_offset - data_chunks[idx][0]
                data = data_chunks[idx][1]
                data = data[sub_offset:sub_offset + cur_size]
            else:
                data = ''
            if len(data) != cur_size:
                raise AssertionError('We must have miscalulated. We expected %d bytes, but only found %d' % (cur_size, len(data)))
            yield (cur_offset, data)
            try:
                cur_offset, cur_size = next(offset_iter)
            except StopIteration:
                return