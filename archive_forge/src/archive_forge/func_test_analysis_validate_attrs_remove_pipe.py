import pytest
from mock import Mock
from spacy.language import Language
from spacy.pipe_analysis import get_attr_info, validate_attrs
def test_analysis_validate_attrs_remove_pipe():
    """Test that attributes are validated correctly on remove."""

    @Language.component('pipe_analysis_c6', assigns=['token.tag'])
    def c1(doc):
        return doc

    @Language.component('pipe_analysis_c7', requires=['token.pos'])
    def c2(doc):
        return doc
    nlp = Language()
    nlp.add_pipe('pipe_analysis_c6')
    nlp.add_pipe('pipe_analysis_c7')
    problems = nlp.analyze_pipes()['problems']
    assert problems['pipe_analysis_c7'] == ['token.pos']
    nlp.remove_pipe('pipe_analysis_c7')
    problems = nlp.analyze_pipes()['problems']
    assert all((p == [] for p in problems.values()))