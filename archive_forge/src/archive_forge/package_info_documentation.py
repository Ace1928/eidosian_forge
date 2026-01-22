import argparse
import platform
import sys
import os
import numpy
import scipy
import gensim
Get the versions of Gensim and its dependencies,
    the location where Gensim is installed and platform on which the system is running.

    Returns
    -------
    dict of (str, str)
        Dictionary containing the versions of Gensim, Python, NumPy, SciPy and platform information.

    