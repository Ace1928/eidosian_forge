from io import BytesIO
import numpy as np
import pandas
from pandas.io.common import stringify_path
from modin.config import NPartitions
from modin.core.io.file_dispatcher import OpenFile
from modin.core.io.text.text_file_dispatcher import TextFileDispatcher

        Read data from `path_or_buf` according to the passed `read_json` `kwargs` parameters.

        Parameters
        ----------
        path_or_buf : str, path object or file-like object
            `path_or_buf` parameter of `read_json` function.
        **kwargs : dict
            Parameters of `read_json` function.

        Returns
        -------
        BaseQueryCompiler
            Query compiler with imported data for further processing.
        