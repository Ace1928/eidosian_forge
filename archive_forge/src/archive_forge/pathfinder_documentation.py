from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from cmdstanpy.cmdstan_args import Method
from cmdstanpy.stanfit.metadata import InferenceMetadata
from cmdstanpy.stanfit.runset import RunSet
from cmdstanpy.utils.stancsv import scan_generic_csv

        Move output CSV files to specified directory.  If files were
        written to the temporary session directory, clean filename.
        E.g., save 'bernoulli-201912081451-1-5nm6as7u.csv' as
        'bernoulli-201912081451-1.csv'.

        :param dir: directory path

        See Also
        --------
        stanfit.RunSet.save_csvfiles
        cmdstanpy.from_csv
        