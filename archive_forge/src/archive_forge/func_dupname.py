import sys
import os
import re
import warnings
import types
import unicodedata
def dupname(node, name):
    node['dupnames'].append(name)
    node['names'].remove(name)
    node.referenced = 1