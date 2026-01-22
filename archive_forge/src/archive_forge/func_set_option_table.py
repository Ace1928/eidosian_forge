import sys, string, re
import getopt
from distutils.errors import *
def set_option_table(self, option_table):
    self.option_table = option_table
    self._build_index()