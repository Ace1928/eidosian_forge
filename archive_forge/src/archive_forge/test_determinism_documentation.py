import os
import subprocess
import sys
import pytest
import matplotlib as mpl
import matplotlib.testing.compare
from matplotlib import pyplot as plt
from matplotlib.testing._markers import needs_ghostscript, needs_usetex

    Test SOURCE_DATE_EPOCH support. Output a document with the environment
    variable SOURCE_DATE_EPOCH set to 2000-01-01 00:00 UTC and check that the
    document contains the timestamp that corresponds to this date (given as an
    argument).

    Parameters
    ----------
    fmt : {"pdf", "ps", "svg"}
        Output format.
    string : bytes
        Timestamp string for 2000-01-01 00:00 UTC.
    