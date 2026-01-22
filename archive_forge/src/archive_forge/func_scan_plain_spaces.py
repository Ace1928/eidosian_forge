from .error import MarkedYAMLError
from .tokens import *
def scan_plain_spaces(self, indent, start_mark):
    chunks = []
    length = 0
    while self.peek(length) in ' ':
        length += 1
    whitespaces = self.prefix(length)
    self.forward(length)
    ch = self.peek()
    if ch in '\r\n\x85\u2028\u2029':
        line_break = self.scan_line_break()
        self.allow_simple_key = True
        prefix = self.prefix(3)
        if (prefix == '---' or prefix == '...') and self.peek(3) in '\x00 \t\r\n\x85\u2028\u2029':
            return
        breaks = []
        while self.peek() in ' \r\n\x85\u2028\u2029':
            if self.peek() == ' ':
                self.forward()
            else:
                breaks.append(self.scan_line_break())
                prefix = self.prefix(3)
                if (prefix == '---' or prefix == '...') and self.peek(3) in '\x00 \t\r\n\x85\u2028\u2029':
                    return
        if line_break != '\n':
            chunks.append(line_break)
        elif not breaks:
            chunks.append(' ')
        chunks.extend(breaks)
    elif whitespaces:
        chunks.append(whitespaces)
    return chunks