from .. import tests, utextwrap
def test_cut(self):
    s = _str_SD
    self.check_cut(s, 0, 0)
    self.check_cut(s, 1, 1)
    self.check_cut(s, 5, 5)
    self.check_cut(s, 6, 5)
    self.check_cut(s, 7, 6)
    self.check_cut(s, 12, 8)
    self.check_cut(s, 13, 9)
    self.check_cut(s, 14, 9)
    self.check_cut('A' * 5, 3, 3)