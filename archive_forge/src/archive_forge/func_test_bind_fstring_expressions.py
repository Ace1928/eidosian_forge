import unittest
import os
from weakref import proxy
from functools import partial
from textwrap import dedent
def test_bind_fstring_expressions(self):
    from kivy.lang import Builder
    root = Builder.load_string(dedent('\n        FloatLayout:\n            Label:\n                id: original\n                text: \'perfect\'\n            Label:\n                id: target1\n                text: f"{\' \'.join(p.upper() for p in original.text)}"\n            Label:\n                id: target2\n                text: f"{\'\'.join(sorted({p.upper() for p in original.text}))}"\n            Label:\n                id: target3\n                text: f"{\'odd\' if len(original.text) % 2 else \'even\'}"\n            Label:\n                id: target4\n                text: f"{original.text[len(original.text) // 2:]}"\n            Label:\n                id: target5\n                text: f"{not len(original.text) % 2}"\n            Label:\n                id: target6\n                text: f"{original.text}" + " some text"\n        '))
    assert root.ids.target1.text == 'P E R F E C T'
    assert root.ids.target2.text == 'CEFPRT'
    assert root.ids.target3.text == 'odd'
    assert root.ids.target4.text == 'fect'
    assert root.ids.target5.text == 'False'
    assert root.ids.target6.text == 'perfect some text'
    root.ids.original.text = 'new text'
    assert root.ids.target1.text == 'N E W   T E X T'
    assert root.ids.target2.text == ' ENTWX'
    assert root.ids.target3.text == 'even'
    assert root.ids.target4.text == 'text'
    assert root.ids.target5.text == 'True'
    assert root.ids.target6.text == 'new text some text'