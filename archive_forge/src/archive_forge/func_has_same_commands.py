from __future__ import absolute_import, division, print_function
import re
def has_same_commands(self, interface):
    len1 = len(self.commands)
    len2 = len(interface.commands)
    return len1 == len2 and len1 == len(frozenset(self.commands).intersection(interface.commands))