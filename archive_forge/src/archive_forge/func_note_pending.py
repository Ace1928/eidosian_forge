import sys
import os
import re
import warnings
import types
import unicodedata
def note_pending(self, pending, priority=None):
    self.transformer.add_pending(pending, priority)