import re
import math
def clone_empty(self):
    line = OutputLine(self.__parent)
    line.set_indent(self.__indent_count, self.__alignment_count)
    return line