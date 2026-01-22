import os
import sys
from textwrap import dedent
from twisted.persisted import sob
from twisted.persisted.styles import Ephemeral
from twisted.python import components
from twisted.trial import unittest
def testStylesBeingSet(self):
    o = Dummy()
    o.foo = 5
    o.setComponent(sob.IPersistable, sob.Persistent(o, 'lala'))
    for style in 'source pickle'.split():
        sob.IPersistable(o).setStyle(style)
        sob.IPersistable(o).save(filename='lala.' + style)
        o1 = sob.load('lala.' + style, style)
        self.assertEqual(o.foo, o1.foo)
        self.assertEqual(sob.IPersistable(o1).style, style)