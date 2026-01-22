import pandas
import pandas._libs.lib as lib
from sqlalchemy import MetaData, Table, create_engine, inspect, text
from modin.core.storage_formats.pandas.parsers import _split_result_for_readers
def get_query_columns(engine, query):
    """
    Extract columns names and python types from the `query`.

    Parameters
    ----------
    engine : sqlalchemy.engine.base.Engine
        SQLAlchemy connection engine.
    query : str
        SQL query.

    Returns
    -------
    dict
        Dictionary with columns names and python types.
    """
    con = engine.connect()
    result = con.execute(text(query))
    cols_names = list(result.keys())
    values = list(result.first())
    cols = dict()
    for i in range(len(cols_names)):
        cols[cols_names[i]] = type(values[i]).__name__
    return cols