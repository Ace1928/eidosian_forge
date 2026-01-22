import sys
import os
import re
import warnings
import types
import unicodedata
def note_indirect_target(self, target):
    self.indirect_targets.append(target)
    if target['names']:
        self.note_refname(target)