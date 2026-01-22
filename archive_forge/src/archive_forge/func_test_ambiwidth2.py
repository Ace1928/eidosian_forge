from .. import tests, utextwrap
def test_ambiwidth2(self):
    w = utextwrap.UTextWrapper(4, ambiguous_width=2)
    s = self._cyrill_char * 8
    self.assertEqual([self._cyrill_char * 2] * 4, w.wrap(s))