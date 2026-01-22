import glob
import warnings
import pandas
from pandas.io.common import stringify_path
from modin.config import NPartitions
from modin.core.io.file_dispatcher import FileDispatcher
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler

        When `*` is in the filename, all partitions are written to their own separate file.

        The filenames is determined as follows:
        - if `*` is in the filename, then it will be replaced by the ascending sequence 0, 1, 2, …
        - if `*` is not in the filename, then the default implementation will be used.

        Parameters
        ----------
        qc : BaseQueryCompiler
            The query compiler of the Modin dataframe that we want
            to run ``to_<format>_glob`` on.
        **kwargs : dict
            Parameters for ``pandas.to_<format>(**kwargs)``.
        