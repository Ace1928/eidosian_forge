import collections
from pathlib import Path
import string
from urllib.request import urlopen
import warnings
from cartopy import config
def validate_projection(self, projection):
    return self._source.validate_projection(projection)