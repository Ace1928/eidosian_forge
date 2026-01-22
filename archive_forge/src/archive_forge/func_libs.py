import sys
import re
import os
from configparser import RawConfigParser
def libs(self, section='default'):
    val = self.vars.interpolate(self._sections[section]['libs'])
    return _escape_backslash(val)