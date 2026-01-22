import os
import sys
import re
def paren_repl(obj):
    torep = obj.group(1)
    numrep = obj.group(2)
    return ','.join([torep] * int(numrep))