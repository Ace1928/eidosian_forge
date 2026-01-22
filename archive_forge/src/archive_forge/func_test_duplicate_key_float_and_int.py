from pyflakes import messages as m
from pyflakes.test.harness import TestCase
def test_duplicate_key_float_and_int(self):
    """
        These do look like different values, but when it comes to their use as
        keys, they compare as equal and so are actually duplicates.
        The literal dict {1: 1, 1.0: 1} actually becomes {1.0: 1}.
        """
    self.flakes('\n            {1: 1, 1.0: 2}\n            ', m.MultiValueRepeatedKeyLiteral, m.MultiValueRepeatedKeyLiteral)