def removeLeadingBlanks(lines):
    ret = []
    for line in lines:
        if ret or line.strip():
            ret.append(line)
    return ret