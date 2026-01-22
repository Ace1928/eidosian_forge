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
def parse_drs(self):
    self.assertToken(self.token(), '(')
    self.assertToken(self.token(), '[')
    refs = set()
    while self.token(0) != ']':
        indices = self._parse_index_list()
        refs.add(self.parse_variable())
        if self.token(0) == ',':
            self.token()
    self.token()
    self.assertToken(self.token(), ',')
    self.assertToken(self.token(), '[')
    conds = []
    while self.token(0) != ']':
        indices = self._parse_index_list()
        conds.extend(self.parse_condition(indices))
        if self.token(0) == ',':
            self.token()
    self.token()
    self.assertToken(self.token(), ')')
    return BoxerDrs(list(refs), conds)