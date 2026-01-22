from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_namespaces(self) -> None:
    s = '\n        <x xmlns="base">\n        <y />\n        <y q="1" x:q="2" y:q="3" />\n        <y:y xml:space="1">here is    some space </y:y>\n        <y:y />\n        <x:y />\n        </x>\n        '
    d = microdom.parseString(s)
    s2 = d.toprettyxml()
    self.assertEqual(d.documentElement.namespace, 'base')
    self.assertEqual(d.documentElement.getElementsByTagName('y')[0].namespace, 'base')
    self.assertEqual(d.documentElement.getElementsByTagName('y')[1].getAttributeNS('base', 'q'), '1')
    d2 = microdom.parseString(s2)
    self.assertEqual(d2.documentElement.namespace, 'base')
    self.assertEqual(d2.documentElement.getElementsByTagName('y')[0].namespace, 'base')
    self.assertEqual(d2.documentElement.getElementsByTagName('y')[1].getAttributeNS('base', 'q'), '1')