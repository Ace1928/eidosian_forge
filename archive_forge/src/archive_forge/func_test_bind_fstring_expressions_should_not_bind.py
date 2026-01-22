import unittest
import os
from weakref import proxy
from functools import partial
from textwrap import dedent
def test_bind_fstring_expressions_should_not_bind(self):
    from kivy.lang import Builder
    root = Builder.load_string(dedent('\n        FloatLayout:\n            Label:\n                id: original\n                text: \'perfect\'\n            Label:\n                id: target1\n                text: f"{\' \'.join([original.text for _ in range(2)])}"\n            Label:\n                id: target2\n                text: f"{original.text.upper()}"\n            Label:\n                id: target3\n                text: f"{sum(obj.width for obj in root.children)}"\n        '))
    assert root.ids.target1.text == ' '
    assert root.ids.target2.text == ''
    assert root.ids.target3.text == '400'
    root.ids.original.text = 'new text'
    root.ids.original.width = 0
    assert root.ids.target1.text == ' '
    assert root.ids.target2.text == ''
    assert root.ids.target3.text == '400'