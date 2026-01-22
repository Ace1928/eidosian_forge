from io import StringIO
def escapeWhitespace(s: str, escapeSpaces: bool):
    with StringIO() as buf:
        for c in s:
            if c == ' ' and escapeSpaces:
                buf.write('·')
            elif c == '\t':
                buf.write('\\t')
            elif c == '\n':
                buf.write('\\n')
            elif c == '\r':
                buf.write('\\r')
            else:
                buf.write(c)
        return buf.getvalue()