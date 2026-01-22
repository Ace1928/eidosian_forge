from .. import tests, utextwrap
def test_fill_indent_without_breaks(self):
    w = utextwrap.UTextWrapper(8, initial_indent=' ' * 4, subsequent_indent=' ' * 4)
    w.break_long_words = False
    self.assertEqual('\n'.join(['    hello', '    ' + _str_D[:2], '    ' + _str_D[2:]]), w.fill(_str_SD))