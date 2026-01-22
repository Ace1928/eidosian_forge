import sys, os
import textwrap
def check_choice(option, opt, value):
    if value in option.choices:
        return value
    else:
        choices = ', '.join(map(repr, option.choices))
        raise OptionValueError(_('option %s: invalid choice: %r (choose from %s)') % (opt, value, choices))