from __future__ import with_statement
import inspect
import keyword
import os
import re
import sys
import time
import tokenize
import warnings
from fnmatch import fnmatch
from optparse import OptionParser
def is_binary_operator(token_type, text):
    return (token_type == tokenize.OP or text in ['and', 'or']) and text not in '()[]{},:.;@=%~'