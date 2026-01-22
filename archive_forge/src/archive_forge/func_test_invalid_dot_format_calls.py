from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_invalid_dot_format_calls(self):
    self.flakes("\n            '{'.format(1)\n        ", m.StringDotFormatInvalidFormat)
    self.flakes("\n            '{} {1}'.format(1, 2)\n        ", m.StringDotFormatMixingAutomatic)
    self.flakes("\n            '{0} {}'.format(1, 2)\n        ", m.StringDotFormatMixingAutomatic)
    self.flakes("\n            '{}'.format(1, 2)\n        ", m.StringDotFormatExtraPositionalArguments)
    self.flakes("\n            '{}'.format(1, bar=2)\n        ", m.StringDotFormatExtraNamedArguments)
    self.flakes("\n            '{} {}'.format(1)\n        ", m.StringDotFormatMissingArgument)
    self.flakes("\n            '{2}'.format()\n        ", m.StringDotFormatMissingArgument)
    self.flakes("\n            '{bar}'.format()\n        ", m.StringDotFormatMissingArgument)
    self.flakes("\n            '{:{:{}}}'.format(1, 2, 3)\n        ", m.StringDotFormatInvalidFormat)
    self.flakes("'{.__class__}'.format('')")
    self.flakes("'{foo[bar]}'.format(foo={'bar': 'barv'})")
    self.flakes("\n            print('{:{}} {}'.format(1, 15, 2))\n        ")
    self.flakes("\n            print('{:2}'.format(1))\n        ")
    self.flakes("\n            '{foo}-{}'.format(1, foo=2)\n        ")
    self.flakes('\n            a = ()\n            "{}".format(*a)\n        ')
    self.flakes('\n            k = {}\n            "{foo}".format(**k)\n        ')