def go_out_grouping(self, s, min, max):
    while self.cursor < self.limit:
        ch = ord(self.current[self.cursor])
        if ch <= max and ch >= min:
            ch -= min
            if s[ch >> 3] & 1 << (ch & 7):
                return True
        self.cursor += 1
    return False