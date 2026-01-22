from collections import Counter
import os
import re
import sys
import traceback
import warnings
from .builder import (
from .dammit import UnicodeDammit
from .element import (
def new_string(self, s, subclass=None):
    """Create a new NavigableString associated with this BeautifulSoup
        object.
        """
    container = self.string_container(subclass)
    return container(s)