import re
import unittest
import jsbeautifier
def test_function_indent(self):
    test_fragment = self.decodesto
    self.options.indent_with_tabs = 1
    self.options.keep_function_indentation = 1
    test_fragment('var foo = function(){ bar() }();', 'var foo = function() {\n\tbar()\n}();')
    self.options.tabs = 1
    self.options.keep_function_indentation = 0
    test_fragment('var foo = function(){ baz() }();', 'var foo = function() {\n\tbaz()\n}();')