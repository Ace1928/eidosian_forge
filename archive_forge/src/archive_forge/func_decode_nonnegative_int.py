import re
import string
def decode_nonnegative_int(s):
    return sum((letter_to_int[a] << 6 * i for i, a in enumerate(s)))