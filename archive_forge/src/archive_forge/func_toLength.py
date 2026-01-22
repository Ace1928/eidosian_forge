def toLength(s):
    """convert a string to  a length"""
    try:
        if s[-2:] == 'cm':
            return float(s[:-2]) * cm
        if s[-2:] == 'in':
            return float(s[:-2]) * inch
        if s[-2:] == 'pt':
            return float(s[:-2])
        if s[-1:] == 'i':
            return float(s[:-1]) * inch
        if s[-2:] == 'mm':
            return float(s[:-2]) * mm
        if s[-4:] == 'pica':
            return float(s[:-4]) * pica
        return float(s)
    except:
        raise ValueError("Can't convert '%s' to length" % s)