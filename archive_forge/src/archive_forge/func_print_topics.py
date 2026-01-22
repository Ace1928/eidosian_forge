import string, sys
def print_topics(self, header, cmds, cmdlen, maxcol):
    if cmds:
        self.stdout.write('%s\n' % str(header))
        if self.ruler:
            self.stdout.write('%s\n' % str(self.ruler * len(header)))
        self.columnize(cmds, maxcol - 1)
        self.stdout.write('\n')