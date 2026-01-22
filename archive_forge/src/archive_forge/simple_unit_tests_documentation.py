from __future__ import division
import pyparsing as pp
from collections import namedtuple
from datetime import datetime

    Base pyparsing testing class to parse various pyparsing expressions against
    given text strings. Subclasses must define a class attribute 'tests' which
    is a list of PpTestSpec instances.
    