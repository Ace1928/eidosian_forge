import typing as T
import pytest
from docstring_parser.numpydoc import compose, parse
def test_other_params() -> None:
    """Test parsing other parameters."""
    docstring = parse('\n        Short description\n        Other Parameters\n        ----------------\n        only_seldom_used_keywords : type, optional\n            Explanation\n        common_parameters_listed_above : type, optional\n            Explanation\n        ')
    assert len(docstring.meta) == 2
    assert docstring.meta[0].args == ['other_param', 'only_seldom_used_keywords']
    assert docstring.meta[0].arg_name == 'only_seldom_used_keywords'
    assert docstring.meta[0].type_name == 'type'
    assert docstring.meta[0].is_optional
    assert docstring.meta[0].description == 'Explanation'
    assert docstring.meta[1].args == ['other_param', 'common_parameters_listed_above']