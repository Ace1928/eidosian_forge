import re
import unittest
import jsbeautifier
import six
import copy
def reset_options(self):
    true = True
    false = False
    default_options = jsbeautifier.default_options()
    default_options.indent_size = 4
    default_options.indent_char = ' '
    default_options.preserve_newlines = true
    default_options.jslint_happy = false
    default_options.indent_level = 0
    default_options.break_chained_methods = false
    default_options.eol = '\n'
    default_options.indent_size = 4
    default_options.indent_char = ' '
    default_options.preserve_newlines = true
    default_options.jslint_happy = false
    self.options = copy.copy(default_options)