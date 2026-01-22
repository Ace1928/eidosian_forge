import pyparsing as pp
def cvt_subscript_int(s):
    ret = 0
    for c in s[0]:
        ret = ret * 10 + subscript_int_map[c]
    return ret