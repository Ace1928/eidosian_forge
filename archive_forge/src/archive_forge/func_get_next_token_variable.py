import operator
import os
import re
import subprocess
import tempfile
from functools import reduce
from optparse import OptionParser
from nltk.internals import find_binary
from nltk.sem.drt import (
from nltk.sem.logic import (
def get_next_token_variable(self, description):
    try:
        return self.token()
    except ExpectedMoreTokensException as e:
        raise ExpectedMoreTokensException(e.index, 'Variable expected.') from e