import os
import sys
import re
def find_repl_patterns(astr):
    reps = named_re.findall(astr)
    names = {}
    for rep in reps:
        name = rep[0].strip() or unique_key(names)
        repl = rep[1].replace('\\,', '@comma@')
        thelist = conv(repl)
        names[name] = thelist
    return names