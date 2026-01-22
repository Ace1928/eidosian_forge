from itertools import chain
from typing import Any, Callable, Iterable, Optional
import numpy
from numpy.typing import ArrayLike
from numpy.typing import DTypeLike
import cupy
from cupy._core.core import ndarray
import cupy._creation.from_data as _creation_from_data
import cupy._core._routines_math as _math
import cupy._core._routines_statistics as _statistics
from cupy.cuda.device import Device
from cupy.cuda.stream import Stream
from cupy.cuda.stream import get_current_stream
from cupyx.distributed.array import _chunk
from cupyx.distributed.array._chunk import _Chunk
from cupyx.distributed.array import _data_transfer
from cupyx.distributed.array._data_transfer import _Communicator
from cupyx.distributed.array import _elementwise
from cupyx.distributed.array import _index_arith
from cupyx.distributed.array import _modes
from cupyx.distributed.array import _reduction
from cupyx.distributed.array import _linalg
def reshard(self, index_map: dict[int, Any]) -> 'DistributedArray':
    """Return a view or a copy having the given index_map.

        Data transfers across devices are done on separate streams created
        internally. To make them asynchronous, transferred data is buffered and
        reflected to the chunks when necessary.

        Args:
            index_map (dict from int to array indices): Indices for the chunks
                that devices with designated IDs own. The current index_map of
                a distributed array can be obtained from
                :attr:`DistributedArray.index_map`.
        """
    new_index_map = _index_arith._normalize_index_map(self.shape, index_map)
    if new_index_map == self.index_map:
        return self
    old_chunks_map = self._chunks_map
    new_chunks_map: dict[int, list[_Chunk]] = {}
    for dev, idxs in new_index_map.items():
        new_chunks_map[dev] = []
        for idx in idxs:
            with Device(dev):
                dst_shape = _index_arith._shape_after_indexing(self.shape, idx)
                new_chunk = _Chunk.create_placeholder(dst_shape, dev, idx)
                new_chunks_map[dev].append(new_chunk)
    self._prepare_comms_and_streams(index_map.keys())
    for src_chunk in chain.from_iterable(old_chunks_map.values()):
        src_chunk.flush(self._mode)
        if self._mode is not _modes.REPLICA:
            src_chunk = src_chunk.copy()
        for dst_chunk in chain.from_iterable(new_chunks_map.values()):
            src_chunk.apply_to(dst_chunk, self._mode, self.shape, self._comms, self._streams)
    return DistributedArray(self.shape, self.dtype, new_chunks_map, self._mode, self._comms)