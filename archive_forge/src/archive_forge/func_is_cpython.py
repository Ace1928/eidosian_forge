import sys
import os
@staticmethod
def is_cpython():
    """
        Suppress supplying distutils for CPython (build and tests).
        Ref #2965 and #3007.
        """
    return os.path.isfile('pybuilddir.txt')