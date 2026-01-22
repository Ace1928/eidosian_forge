import re
import string
def supress_minus_zero(x):
    return 0 if x == 0 else x