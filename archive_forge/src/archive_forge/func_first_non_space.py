import sys
import tokenize
def first_non_space(s):
    for i, c in enumerate(s):
        if c != ' ':
            return i
    return 0