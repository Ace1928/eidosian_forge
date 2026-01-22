import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
def isAllowedName(self, node, name):
    if name is None:
        return
    self.nameIsAllowed(name)