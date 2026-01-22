from pyflakes import messages as m
from pyflakes.test.harness import TestCase

        These do look like different values, but when it comes to their use as
        keys, they compare as equal and so are actually duplicates.
        The literal dict {1: 1, 1.0: 1} actually becomes {1.0: 1}.
        