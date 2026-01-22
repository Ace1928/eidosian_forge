import re
from pygments.lexer import RegexLexer, ExtendedRegexLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import shebang_matches
def strip_pod(lines):
    in_pod = False
    stripped_lines = []
    for line in lines:
        if re.match('^=(?:end|cut)', line):
            in_pod = False
        elif re.match('^=\\w+', line):
            in_pod = True
        elif not in_pod:
            stripped_lines.append(line)
    return stripped_lines