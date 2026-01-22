import warnings
import numpy as np
import pandas
from modin.config import NPartitions
from modin.core.io import SQLDispatcher

        Read SQL query or database table into a DataFrame.

        Documentation for parameters can be found at `modin.read_sql`.

        Returns
        -------
        BaseQueryCompiler
            A new query compiler with imported data for further processing.
        