def eq_s_b(self, s):
    if self.cursor - self.limit_backward < len(s):
        return False
    if self.current[self.cursor - len(s):self.cursor] != s:
        return False
    self.cursor -= len(s)
    return True