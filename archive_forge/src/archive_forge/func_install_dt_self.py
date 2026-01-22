from lxml import etree
import sys
import re
import doctest
def install_dt_self(self):
    self.prev_func = self.dt_self._DocTestRunner__record_outcome
    self.dt_self._DocTestRunner__record_outcome = self