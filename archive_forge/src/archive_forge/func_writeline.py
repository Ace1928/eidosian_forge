import re
from mako import exceptions
def writeline(self, line):
    """print a line of python, indenting it according to the current
        indent level.

        this also adjusts the indentation counter according to the
        content of the line.

        """
    if not self.in_indent_lines:
        self._flush_adjusted_lines()
        self.in_indent_lines = True
    if line is None or self._re_space_comment.match(line) or self._re_space.match(line):
        hastext = False
    else:
        hastext = True
    is_comment = line and len(line) and (line[0] == '#')
    if not is_comment and (not hastext or self._is_unindentor(line)) and (self.indent > 0):
        self.indent -= 1
        if len(self.indent_detail) == 0:
            raise exceptions.MakoException('Too many whitespace closures')
        self.indent_detail.pop()
    if line is None:
        return
    self.stream.write(self._indent_line(line) + '\n')
    self._update_lineno(len(line.split('\n')))
    if self._re_indent.search(line):
        match = self._re_compound.match(line)
        if match:
            indentor = match.group(1)
            self.indent += 1
            self.indent_detail.append(indentor)
        else:
            indentor = None
            m2 = self._re_indent_keyword.match(line)
            if m2:
                self.indent += 1
                self.indent_detail.append(indentor)