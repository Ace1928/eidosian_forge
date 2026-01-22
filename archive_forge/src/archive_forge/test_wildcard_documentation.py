import unittest
from IPython.utils import wildcard
Dictionaries should be indexed by attributes, not by keys. This was
        causing Github issue 129.