def slice_check(self):
    if self.bra < 0 or self.bra > self.ket or self.ket > self.limit or (self.limit > len(self.current)):
        return False
    return True