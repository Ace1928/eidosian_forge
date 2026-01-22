from .. import tests, utextwrap
def test_fill_indent_with_breaks(self):
    w = utextwrap.UTextWrapper(8, initial_indent=' ' * 4, subsequent_indent=' ' * 4)
    self.assertEqual('\n'.join(['    hell', '    o' + _str_D[0], '    ' + _str_D[1:3], '    ' + _str_D[3]]), w.fill(_str_SD))