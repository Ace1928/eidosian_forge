import math
from typing import Any, Generic, Iterable, List, Optional, Tuple, TypeVar
import numpy as np
import pandas as pd
import pyarrow as pa
from triad.utils.convert import to_size
from .iter import slice_iterable
def reslice_and_merge(self, batches: Iterable[T]) -> Iterable[T]:
    """Reslice the batch stream into new batches, each containing the same keys

        :param batches: the batch stream

        :yield: an iterable of batches, each containing the same keys
        """
    cache: Optional[T] = None
    for batch in batches:
        if self.get_batch_length(batch) > 0:
            for diff, sub in self._reslice_single(batch):
                if not diff:
                    cache = self.concat([cache, sub])
                else:
                    if cache is not None:
                        yield cache
                    cache = sub
    if cache is not None:
        yield cache