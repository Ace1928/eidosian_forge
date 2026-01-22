from __future__ import annotations
import re
from collections import defaultdict
import numpy as np
def read_matrix_pattern(header_pattern, footer_pattern, elements_pattern, text, postprocess=str):
    """Parse a matrix to get the quantities in a numpy array."""
    header_regex = re.compile(header_pattern)
    footer_regex = re.compile(footer_pattern)
    text_between_header_and_footer = text[header_regex.search(text).end():footer_regex.search(text).start()]
    elements = re.findall(elements_pattern, text_between_header_and_footer)
    return [postprocess(elem) for elem in elements]