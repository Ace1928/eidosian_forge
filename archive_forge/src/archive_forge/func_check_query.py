import pandas
import pandas._libs.lib as lib
from sqlalchemy import MetaData, Table, create_engine, inspect, text
from modin.core.storage_formats.pandas.parsers import _split_result_for_readers
def check_query(query):
    """
    Check query sanity.

    Parameters
    ----------
    query : str
        Query string.
    """
    q = query.lower()
    if 'select ' not in q:
        raise InvalidQuery('SELECT word not found in the query: {0}'.format(query))
    if ' from ' not in q:
        raise InvalidQuery('FROM word not found in the query: {0}'.format(query))