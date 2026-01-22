import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite
def test_node_type_processing_instruction(self):
    xml = XML('<?python x = 2 * 3 ?><root><?php echo("x") ?></root>')
    self._test_eval(path='//processing-instruction()', equiv='<Path "descendant-or-self::processing-instruction()">', input=xml, output='<?python x = 2 * 3 ?><?php echo("x") ?>')
    self._test_eval(path='processing-instruction()', equiv='<Path "child::processing-instruction()">', input=xml, output='<?php echo("x") ?>')
    self._test_eval(path='processing-instruction("php")', equiv='<Path "child::processing-instruction("php")">', input=xml, output='<?php echo("x") ?>')