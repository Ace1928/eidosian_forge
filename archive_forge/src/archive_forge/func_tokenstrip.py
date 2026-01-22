import sys
import re
import copy
import time
import os.path
def tokenstrip(self, tokens):
    i = 0
    while i < len(tokens) and tokens[i].type in self.t_WS:
        i += 1
    del tokens[:i]
    i = len(tokens) - 1
    while i >= 0 and tokens[i].type in self.t_WS:
        i -= 1
    del tokens[i + 1:]
    return tokens