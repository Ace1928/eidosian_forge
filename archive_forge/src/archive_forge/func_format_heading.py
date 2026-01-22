import sys, os
import textwrap
def format_heading(self, heading):
    return '%s\n%s\n' % (heading, '=-'[self.level] * len(heading))