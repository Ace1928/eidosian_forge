import pandas
import pandas._libs.lib as lib
from sqlalchemy import MetaData, Table, create_engine, inspect, text
from modin.core.storage_formats.pandas.parsers import _split_result_for_readers
def query_put_bounders(query, partition_column, start, end):
    """
    Put partition boundaries into the query.

    Parameters
    ----------
    query : str
        SQL query string.
    partition_column : str
        Column name used for data partitioning between the workers.
    start : int
        Lowest value to request from the `partition_column`.
    end : int
        Highest value to request from the `partition_column`.

    Returns
    -------
    str
        Query string with boundaries.
    """
    where = ' WHERE TMP_TABLE.{0} >= {1} AND TMP_TABLE.{0} <= {2}'.format(partition_column, start, end)
    query_with_bounders = 'SELECT * FROM ({0}) AS TMP_TABLE {1}'.format(query, where)
    return query_with_bounders