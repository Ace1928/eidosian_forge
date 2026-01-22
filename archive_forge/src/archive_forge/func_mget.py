from typing import Dict, List, Optional, Tuple, Union
from redis.exceptions import DataError
from redis.typing import KeyT, Number
def mget(self, filters: List[str], with_labels: Optional[bool]=False, select_labels: Optional[List[str]]=None, latest: Optional[bool]=False):
    """# noqa
        Get the last samples matching the specific `filter`.

        Args:

        filters:
            Filter to match the time-series labels.
        with_labels:
            Include in the reply all label-value pairs representing metadata
            labels of the time series.
        select_labels:
            Include in the reply only a subset of the key-value pair labels of a series.
        latest:
            Used when a time series is a compaction, reports the compacted
            value of the latest possibly partial bucket

        For more information: https://redis.io/commands/ts.mget/
        """
    params = []
    self._append_latest(params, latest)
    self._append_with_labels(params, with_labels, select_labels)
    params.extend(['FILTER'])
    params += filters
    return self.execute_command(MGET_CMD, *params)