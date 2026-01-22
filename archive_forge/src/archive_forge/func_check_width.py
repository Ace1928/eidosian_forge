from .. import tests, utextwrap
def check_width(self, text, expected_width):
    w = utextwrap.UTextWrapper()
    self.assertEqual(w._width(text), expected_width, 'Width of %r should be %d' % (text, expected_width))