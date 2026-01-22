from __future__ import absolute_import, division, print_function
import time
def hyphen_to_underline(input):
    if input and isinstance(input, str):
        return input.replace('-', '_')
    return input