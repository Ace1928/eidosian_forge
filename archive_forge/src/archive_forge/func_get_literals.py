import re
import sys
import types
import copy
import os
import inspect
def get_literals(self):
    self.literals = self.ldict.get('literals', '')
    if not self.literals:
        self.literals = ''