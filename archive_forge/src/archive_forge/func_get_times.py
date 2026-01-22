import warnings
import json
import csv
import numpy as np
from Bio import BiopythonParserWarning
def get_times(self):
    """Get a list of the recorded time points."""
    return sorted(self._signals.keys())