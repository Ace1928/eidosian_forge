from contextlib import ExitStack
from io import BytesIO
from multiprocessing.pool import ThreadPool
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm

        Create a reader for part of the CSV.

        Parameters
        ----------
        arg : tuple
            A tuple of the following:

            * start : int
                The starting row to start for parsing CSV
            * nrows : int
                The number of rows to read.

        Returns
        -------
        df : DataFrame
        