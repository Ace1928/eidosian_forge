from __future__ import unicode_literals
import pytest
from six.moves import zip_longest
from pybtex.bibtex import bst
from ..utils import get_data
@pytest.mark.parametrize(['dataset_name'], [('plain',), ('apacite',), ('jurabib',)])
def test_bst_parser(dataset_name):
    module = __import__('tests.bst_parser_test.{0}'.format(dataset_name), globals(), locals(), 'bst')
    correct_result = module.bst
    bst_data = get_data(dataset_name + '.bst')
    actual_result = bst.parse_string(bst_data)
    for correct_element, actual_element in zip_longest(actual_result, correct_result):
        assert correct_element == actual_element, '\n{0}\n{1}'.format(correct_element, actual_element)