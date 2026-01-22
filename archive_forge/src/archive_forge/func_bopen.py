import codecs
import re
import warnings
from typing import Match
def bopen(*args, **kwargs):
    return open(*args, mode='rb', **kwargs)