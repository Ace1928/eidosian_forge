import os
import sys
import re
def parse_loop_header(loophead):
    """Find all named replacements in the header

    Returns a list of dictionaries, one for each loop iteration,
    where each key is a name to be substituted and the corresponding
    value is the replacement string.

    Also return a list of exclusions.  The exclusions are dictionaries
     of key value pairs. There can be more than one exclusion.
     [{'var1':'value1', 'var2', 'value2'[,...]}, ...]

    """
    loophead = stripast.sub('', loophead)
    names = []
    reps = named_re.findall(loophead)
    nsub = None
    for rep in reps:
        name = rep[0]
        vals = parse_values(rep[1])
        size = len(vals)
        if nsub is None:
            nsub = size
        elif nsub != size:
            msg = 'Mismatch in number of values, %d != %d\n%s = %s'
            raise ValueError(msg % (nsub, size, name, vals))
        names.append((name, vals))
    excludes = []
    for obj in exclude_re.finditer(loophead):
        span = obj.span()
        endline = loophead.find('\n', span[1])
        substr = loophead[span[1]:endline]
        ex_names = exclude_vars_re.findall(substr)
        excludes.append(dict(ex_names))
    dlist = []
    if nsub is None:
        raise ValueError('No substitution variables found')
    for i in range(nsub):
        tmp = {name: vals[i] for name, vals in names}
        dlist.append(tmp)
    return dlist