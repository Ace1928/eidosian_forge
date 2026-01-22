from __future__ import absolute_import, unicode_literals
def prefixed_digits(prefix, digits):
    return prefix + Opt(Str('_')) + underscore_digits(digits)