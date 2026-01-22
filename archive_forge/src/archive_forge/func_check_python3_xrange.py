import re
from hacking import core
import pycodestyle
@core.flake8ext
def check_python3_xrange(logical_line):
    if re.search('\\bxrange\\s*\\(', logical_line):
        yield (0, 'D708: Do not use xrange. Use range for large loops.')