import sys, io
def gen_error(self, msg, line=None):
    outmsg = []
    if line is None:
        line = self.current_line
    outmsg.append(self.filename + ', ')
    if isinstance(line, (list, tuple)):
        outmsg.append('lines %d-%d: ' % tuple(line))
    else:
        outmsg.append('line %d: ' % line)
    outmsg.append(str(msg))
    return ''.join(outmsg)