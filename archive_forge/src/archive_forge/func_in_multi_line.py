import re
from mako import exceptions
def in_multi_line(line):
    start_state = state[backslashed] or state[triplequoted]
    if re.search('\\\\$', line):
        state[backslashed] = True
    else:
        state[backslashed] = False

    def match(reg, t):
        m = re.match(reg, t)
        if m:
            return (m, t[len(m.group(0)):])
        else:
            return (None, t)
    while line:
        if state[triplequoted]:
            m, line = match('%s' % state[triplequoted], line)
            if m:
                state[triplequoted] = False
            else:
                m, line = match('.*?(?=%s|$)' % state[triplequoted], line)
        else:
            m, line = match('#', line)
            if m:
                return start_state
            m, line = match('\\"\\"\\"|\\\'\\\'\\\'', line)
            if m:
                state[triplequoted] = m.group(0)
                continue
            m, line = match('.*?(?=\\"\\"\\"|\\\'\\\'\\\'|#|$)', line)
    return start_state