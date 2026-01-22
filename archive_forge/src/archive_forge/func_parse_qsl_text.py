import sys
import types
from cgi import parse_header
def parse_qsl_text(qs, encoding='utf-8'):
    qsl = parse_qsl(qs, keep_blank_values=True, strict_parsing=False)
    for x, y in qsl:
        yield (x.decode(encoding), y.decode(encoding))