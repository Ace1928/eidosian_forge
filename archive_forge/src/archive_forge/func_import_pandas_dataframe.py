import abc
import uuid
from typing import List, Tuple
import numpy as np
import pyarrow as pa
from modin.error_message import ErrorMessage
@classmethod
def import_pandas_dataframe(cls, df, name=None):
    """
        Import ``pandas.DataFrame`` to the worker.

        Parameters
        ----------
        df : pandas.DataFrame
            A frame to import.
        name : str, optional
            A table name to use. None to generate a unique name.

        Returns
        -------
        DbTable
            Imported table.
        """
    return cls.import_arrow_table(pa.Table.from_pandas(df), name=name)