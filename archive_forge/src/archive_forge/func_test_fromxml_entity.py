from __future__ import absolute_import, print_function, division
import sys
from collections import OrderedDict
from tempfile import NamedTemporaryFile
import pytest
from petl.test.helpers import ieq
from petl.util import nrows, look
from petl.io.xml import fromxml, toxml
from petl.compat import urlopen
def test_fromxml_entity():
    _DATA1 = '\n        <tr><td>foo</td><td>bar</td></tr>\n        <tr><td>a</td><td>1</td></tr>\n        <tr><td>b</td><td>2</td></tr>\n        <tr><td>c</td><td>3</td></tr>\n        '
    _DATA2 = '<td>X</td><td>9</td>'
    _DOCTYPE = '<?xml version="1.0" encoding="ISO-8859-1"?>\n    <!DOCTYPE foo [  \n        <!ELEMENT table ANY >\n        <!ENTITY inserted SYSTEM "file://%s" >]>\n        '
    _INSERTED = '<tr>&inserted;</tr>'
    _TABLE1 = (('foo', 'bar'), ('a', '1'), ('b', '2'), ('c', '3'))
    temp_file1 = _write_test_file(_DATA1)
    actual11 = fromxml(temp_file1, 'tr', 'td')
    _compare(_TABLE1, actual11)
    try:
        from lxml import etree
    except:
        return
    data_file_tmp = _write_temp_file(_DATA2)
    doc_type_temp = _DOCTYPE % data_file_tmp
    doc_type_miss = _DOCTYPE % '/tmp/doesnotexist'
    _EXPECT_IT = (('X', '9'),)
    _EXPECT_NO = ((None, None),)
    temp_file2 = _write_test_file(_DATA1, pre=doc_type_temp, pos=_INSERTED)
    temp_file3 = _write_test_file(_DATA1, pre=doc_type_miss, pos=_INSERTED)
    parser_off = etree.XMLParser(resolve_entities=False)
    parser_onn = etree.XMLParser(resolve_entities=True)
    actual12 = fromxml(temp_file1, 'tr', 'td', parser=parser_off)
    _compare(_TABLE1, actual12)
    actual21 = fromxml(temp_file2, 'tr', 'td')
    _compare(_TABLE1 + _EXPECT_NO, actual21)
    actual22 = fromxml(temp_file2, 'tr', 'td', parser=parser_off)
    _compare(_TABLE1 + _EXPECT_NO, actual22)
    actual23 = fromxml(temp_file2, 'tr', 'td', parser=parser_onn)
    _compare(_TABLE1 + _EXPECT_IT, actual23)
    actual31 = fromxml(temp_file3, 'tr', 'td')
    _compare(_TABLE1 + _EXPECT_NO, actual31)
    actual32 = fromxml(temp_file3, 'tr', 'td', parser=parser_off)
    _compare(_TABLE1 + _EXPECT_NO, actual32)
    try:
        actual33 = fromxml(temp_file3, 'tr', 'td', parser=parser_onn)
        for _ in actual33:
            pass
    except etree.XMLSyntaxError:
        pass
    else:
        assert True, 'Error testing XML'