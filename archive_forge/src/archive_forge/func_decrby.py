from typing import Dict, List, Optional, Tuple, Union
from redis.exceptions import DataError
from redis.typing import KeyT, Number
def decrby(self, key: KeyT, value: Number, timestamp: Optional[Union[int, str]]=None, retention_msecs: Optional[int]=None, uncompressed: Optional[bool]=False, labels: Optional[Dict[str, str]]=None, chunk_size: Optional[int]=None):
    """
        Decrement (or create an time-series and decrement) the latest sample's of a series.
        This command can be used as a counter or gauge that automatically gets history as a time series.

        Args:

        key:
            time-series key
        value:
            Numeric data value of the sample
        timestamp:
            Timestamp of the sample. * can be used for automatic timestamp (using the system clock).
        retention_msecs:
            Maximum age for samples compared to last event time (in milliseconds).
            If None or 0 is passed then  the series is not trimmed at all.
        uncompressed:
            Changes data storage from compressed (by default) to uncompressed
        labels:
            Set of label-value pairs that represent metadata labels of the key.
        chunk_size:
            Memory size, in bytes, allocated for each data chunk.

        For more information: https://redis.io/commands/ts.decrby/
        """
    params = [key, value]
    self._append_timestamp(params, timestamp)
    self._append_retention(params, retention_msecs)
    self._append_uncompressed(params, uncompressed)
    self._append_chunk_size(params, chunk_size)
    self._append_labels(params, labels)
    return self.execute_command(DECRBY_CMD, *params)