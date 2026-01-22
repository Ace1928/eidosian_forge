from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_invalid_percent_format_calls(self):
    self.flakes("\n            '%(foo)' % {'foo': 'bar'}\n        ", m.PercentFormatInvalidFormat)
    self.flakes("\n            '%s %(foo)s' % {'foo': 'bar'}\n        ", m.PercentFormatMixedPositionalAndNamed)
    self.flakes("\n            '%(foo)s %s' % {'foo': 'bar'}\n        ", m.PercentFormatMixedPositionalAndNamed)
    self.flakes("\n            '%j' % (1,)\n        ", m.PercentFormatUnsupportedFormatCharacter)
    self.flakes("\n            '%s %s' % (1,)\n        ", m.PercentFormatPositionalCountMismatch)
    self.flakes("\n            '%s %s' % (1, 2, 3)\n        ", m.PercentFormatPositionalCountMismatch)
    self.flakes("\n            '%(bar)s' % {}\n        ", m.PercentFormatMissingArgument)
    self.flakes("\n            '%(bar)s' % {'bar': 1, 'baz': 2}\n        ", m.PercentFormatExtraNamedArguments)
    self.flakes("\n            '%(bar)s' % (1, 2, 3)\n        ", m.PercentFormatExpectedMapping)
    self.flakes("\n            '%s %s' % {'k': 'v'}\n        ", m.PercentFormatExpectedSequence)
    self.flakes("\n            '%(bar)*s' % {'bar': 'baz'}\n        ", m.PercentFormatStarRequiresSequence)
    self.flakes("\n            '%s' % {'foo': 'bar', 'baz': 'womp'}\n        ")
    self.flakes('\n            "%1000000000000f" % 1\n        ')
    self.flakes("\n            '%% %s %% %s' % (1, 2)\n        ")
    self.flakes("\n            '%.*f' % (2, 1.1234)\n            '%*.*f' % (5, 2, 3.1234)\n        ")