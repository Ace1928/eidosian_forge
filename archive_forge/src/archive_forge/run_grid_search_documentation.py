import itertools
import os
import uuid
from datetime import date
from pathlib import Path
from typing import Dict, Iterable
import submitit
from xformers.benchmarks.LRA.run_with_submitit import (

    Yield all combinations of parameters in the grid (as a dict)
    