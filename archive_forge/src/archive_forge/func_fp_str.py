import reportlab
def fp_str(*a):
    """convert separate arguments (or single sequence arg) into space separated numeric strings"""
    if len(a) == 1 and isSeq(a[0]):
        a = a[0]
    s = []
    A = s.append
    for i in a:
        sa = abs(i)
        if sa <= 1e-07:
            A('0')
        else:
            l = sa <= 1 and 6 or min(max(0, 6 - int(_log_10(sa))), 6)
            n = _fp_fmts[l] % i
            if l:
                j = len(n)
                while j:
                    j -= 1
                    if n[j] != '0':
                        if n[j] != '.':
                            j += 1
                        break
                n = n[:j]
            A((n[0] != '0' or len(n) == 1) and n or n[1:])
    return ' '.join(s)