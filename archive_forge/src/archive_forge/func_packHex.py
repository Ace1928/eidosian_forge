def packHex(self, n):
    if n in range(48, 58):
        return n - 48
    if n in range(65, 71):
        return 10 + (n - 65)
    return -1