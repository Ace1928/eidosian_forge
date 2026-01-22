import ast
import os
import re
import sys
import breezy.branch
from breezy import osutils
from breezy.tests import TestCase, TestSkipped, features
def find_occurences(self, rule, filename):
    """Find the number of occurences of rule in a file."""
    occurences = 0
    source = open(filename)
    for line in source:
        if line.find(rule) > -1:
            occurences += 1
    return occurences