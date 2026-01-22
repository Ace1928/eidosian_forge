def in_grouping(self, s, min, max):
    if self.cursor >= self.limit:
        return False
    ch = ord(self.current[self.cursor])
    if ch > max or ch < min:
        return False
    ch -= min
    if s[ch >> 3] & 1 << (ch & 7) == 0:
        return False
    self.cursor += 1
    return True