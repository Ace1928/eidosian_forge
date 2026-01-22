import datetime
import platform
import subprocess
from typing import Optional, Tuple, Union
import numpy as np
def chunk_bytes_iter(iterator, chunk_len: int, stride: Tuple[int, int], stream: bool=False):
    """
    Reads raw bytes from an iterator and does chunks of length `chunk_len`. Optionally adds `stride` to each chunks to
    get overlaps. `stream` is used to return partial results even if a full `chunk_len` is not yet available.
    """
    acc = b''
    stride_left, stride_right = stride
    if stride_left + stride_right >= chunk_len:
        raise ValueError(f'Stride needs to be strictly smaller than chunk_len: ({stride_left}, {stride_right}) vs {chunk_len}')
    _stride_left = 0
    for raw in iterator:
        acc += raw
        if stream and len(acc) < chunk_len:
            stride = (_stride_left, 0)
            yield {'raw': acc[:chunk_len], 'stride': stride, 'partial': True}
        else:
            while len(acc) >= chunk_len:
                stride = (_stride_left, stride_right)
                item = {'raw': acc[:chunk_len], 'stride': stride}
                if stream:
                    item['partial'] = False
                yield item
                _stride_left = stride_left
                acc = acc[chunk_len - stride_left - stride_right:]
    if len(acc) > stride_left:
        item = {'raw': acc, 'stride': (_stride_left, 0)}
        if stream:
            item['partial'] = False
        yield item