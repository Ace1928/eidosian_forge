from typing import Dict, List, Optional, Tuple, Union
from redis.exceptions import DataError
from redis.typing import KeyT, Number
def mrevrange(self, from_time: Union[int, str], to_time: Union[int, str], filters: List[str], count: Optional[int]=None, aggregation_type: Optional[str]=None, bucket_size_msec: Optional[int]=0, with_labels: Optional[bool]=False, filter_by_ts: Optional[List[int]]=None, filter_by_min_value: Optional[int]=None, filter_by_max_value: Optional[int]=None, groupby: Optional[str]=None, reduce: Optional[str]=None, select_labels: Optional[List[str]]=None, align: Optional[Union[int, str]]=None, latest: Optional[bool]=False, bucket_timestamp: Optional[str]=None, empty: Optional[bool]=False):
    """
        Query a range across multiple time-series by filters in reverse direction.

        Args:

        from_time:
            Start timestamp for the range query. - can be used to express the minimum possible timestamp (0).
        to_time:
            End timestamp for range query, + can be used to express the maximum possible timestamp.
        filters:
            Filter to match the time-series labels.
        count:
            Limits the number of returned samples.
        aggregation_type:
            Optional aggregation type. Can be one of [`avg`, `sum`, `min`, `max`,
            `range`, `count`, `first`, `last`, `std.p`, `std.s`, `var.p`, `var.s`, `twa`]
        bucket_size_msec:
            Time bucket for aggregation in milliseconds.
        with_labels:
            Include in the reply all label-value pairs representing metadata labels of the time series.
        filter_by_ts:
            List of timestamps to filter the result by specific timestamps.
        filter_by_min_value:
            Filter result by minimum value (must mention also filter_by_max_value).
        filter_by_max_value:
            Filter result by maximum value (must mention also filter_by_min_value).
        groupby:
            Grouping by fields the results (must mention also reduce).
        reduce:
            Applying reducer functions on each group. Can be one of [`avg` `sum`, `min`,
            `max`, `range`, `count`, `std.p`, `std.s`, `var.p`, `var.s`].
        select_labels:
            Include in the reply only a subset of the key-value pair labels of a series.
        align:
            Timestamp for alignment control for aggregation.
        latest:
            Used when a time series is a compaction, reports the compacted
            value of the latest possibly partial bucket
        bucket_timestamp:
            Controls how bucket timestamps are reported. Can be one of [`-`, `low`, `+`,
            `high`, `~`, `mid`].
        empty:
            Reports aggregations for empty buckets.

        For more information: https://redis.io/commands/ts.mrevrange/
        """
    params = self.__mrange_params(aggregation_type, bucket_size_msec, count, filters, from_time, to_time, with_labels, filter_by_ts, filter_by_min_value, filter_by_max_value, groupby, reduce, select_labels, align, latest, bucket_timestamp, empty)
    return self.execute_command(MREVRANGE_CMD, *params)