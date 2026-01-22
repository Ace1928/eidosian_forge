import codecs
import re
from io import StringIO
from xml.etree.ElementTree import Element, ElementTree, SubElement, TreeBuilder
from nltk.data import PathPointer, find
def raw_fields(self):
    """
        Return an iterator that returns the next field in a (marker, value)
        tuple. Linebreaks and trailing white space are preserved except
        for the final newline in each field.

        :rtype: iter(tuple(str, str))
        """
    join_string = '\n'
    line_regexp = '^%s(?:\\\\(\\S+)\\s*)?(.*)$'
    first_line_pat = re.compile(line_regexp % '(?:ï»¿)?')
    line_pat = re.compile(line_regexp % '')
    file_iter = iter(self._file)
    try:
        line = next(file_iter)
    except StopIteration:
        return
    mobj = re.match(first_line_pat, line)
    mkr, line_value = mobj.groups()
    value_lines = [line_value]
    self.line_num = 0
    for line in file_iter:
        self.line_num += 1
        mobj = re.match(line_pat, line)
        line_mkr, line_value = mobj.groups()
        if line_mkr:
            yield (mkr, join_string.join(value_lines))
            mkr = line_mkr
            value_lines = [line_value]
        else:
            value_lines.append(line_value)
    self.line_num += 1
    yield (mkr, join_string.join(value_lines))