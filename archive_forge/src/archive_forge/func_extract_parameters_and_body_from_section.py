import re
from . import utilities
def extract_parameters_and_body_from_section(section_text):
    """
    Turns patterns of the form "==KEY:VALUE" at the beginning lines
    into a dictionary and returns the remaining text.

    >>> t = "==A:1\\n==B:2\\nBody\\nBody"
    >>> extract_parameters_and_body_from_section(t)[0]['A']
    '1'
    """
    params = {}
    while True:
        m = re.match('==(.*):(.*)\n', section_text)
        if not m:
            return (params, section_text)
        k, v = m.groups()
        params[k] = v
        section_text = section_text.split('\n', 1)[1]