def go_in_grouping(self, s, min, max):
    while self.cursor < self.limit:
        ch = ord(self.current[self.cursor])
        if ch > max or ch < min:
            return True
        ch -= min
        if s[ch >> 3] & 1 << (ch & 7) == 0:
            return True
        self.cursor += 1
    return False