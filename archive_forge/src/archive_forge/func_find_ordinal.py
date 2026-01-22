import sys
import typing
from datetime import timedelta
def find_ordinal(pos_num: int) -> str:
    if pos_num == 0:
        return 'th'
    elif pos_num == 1:
        return 'st'
    elif pos_num == 2:
        return 'nd'
    elif pos_num == 3:
        return 'rd'
    elif 4 <= pos_num <= 20:
        return 'th'
    else:
        return find_ordinal(pos_num % 10)