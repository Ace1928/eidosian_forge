from __future__ import unicode_literals
import pytest
from pybtex import errors
from pybtex.database import InvalidNameString, Person
@pytest.mark.parametrize(['name', 'correct_result', 'expected_errors'], sample_names)
def test_parse_name(name, correct_result, expected_errors):
    if expected_errors is None:
        expected_errors = []
    with errors.capture() as captured_errors:
        person = Person(name)
    result = (person.bibtex_first_names, person.prelast_names, person.last_names, person.lineage_names)
    assert result == correct_result
    assert captured_errors == expected_errors