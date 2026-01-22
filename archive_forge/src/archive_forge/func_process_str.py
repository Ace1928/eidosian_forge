import os
import sys
import re
def process_str(astr):
    code = [header]
    code.extend(parse_string(astr, global_names, 0, 1))
    return ''.join(code)