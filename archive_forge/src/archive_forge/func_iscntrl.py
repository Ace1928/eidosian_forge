def iscntrl(c):
    return 0 <= _ctoi(c) <= 31 or _ctoi(c) == 127