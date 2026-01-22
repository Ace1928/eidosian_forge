import pandas
import pandas._libs.lib as lib
from sqlalchemy import MetaData, Table, create_engine, inspect, text
from modin.core.storage_formats.pandas.parsers import _split_result_for_readers
def is_distributed(partition_column, lower_bound, upper_bound):
    """
    Check if is possible to distribute a query with the given args.

    Parameters
    ----------
    partition_column : str
        Column name used for data partitioning between the workers.
    lower_bound : int
        The minimum value to be requested from the `partition_column`.
    upper_bound : int
        The maximum value to be requested from the `partition_column`.

    Returns
    -------
    bool
        Whether the given query is distributable or not.
    """
    if partition_column is not None and lower_bound is not None and (upper_bound is not None):
        if upper_bound > lower_bound:
            return True
        raise InvalidArguments('upper_bound must be greater than lower_bound.')
    elif partition_column is None and lower_bound is None and (upper_bound is None):
        return False
    else:
        raise InvalidArguments('Invalid combination of partition_column, lower_bound, upper_bound.' + 'All these arguments should be passed (distributed) or none of them (standard pandas).')