import os
import sys
import tokenize
def indent_level(self, tabsize):
    count, trailing = self.norm
    il = 0
    for i in range(tabsize, len(count)):
        il = il + i // tabsize * count[i]
    return trailing + tabsize * (il + self.nt)