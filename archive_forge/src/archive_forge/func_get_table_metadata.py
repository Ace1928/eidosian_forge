import pandas
import pandas._libs.lib as lib
from sqlalchemy import MetaData, Table, create_engine, inspect, text
from modin.core.storage_formats.pandas.parsers import _split_result_for_readers
def get_table_metadata(engine, table):
    """
    Extract all useful data from the given table.

    Parameters
    ----------
    engine : sqlalchemy.engine.base.Engine
        SQLAlchemy connection engine.
    table : str
        Table name.

    Returns
    -------
    sqlalchemy.sql.schema.Table
        Extracted metadata.
    """
    metadata = MetaData()
    metadata.reflect(bind=engine, only=[table])
    table_metadata = Table(table, metadata, autoload=True)
    return table_metadata