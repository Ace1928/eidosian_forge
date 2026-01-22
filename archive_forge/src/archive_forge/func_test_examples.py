import typing as T
import pytest
from docstring_parser.numpydoc import compose, parse
@pytest.mark.parametrize('source, expected_results', [('Description\nExamples\n--------\nlong example\n\nmore here', [(None, 'long example\n\nmore here')]), ('Description\nExamples\n--------\n>>> test', [('>>> test', '')]), ('Description\nExamples\n--------\n>>> testa\n>>> testb', [('>>> testa\n>>> testb', '')]), ('Description\nExamples\n--------\n>>> test1\ndesc1', [('>>> test1', 'desc1')]), ('Description\nExamples\n--------\n>>> test1a\n>>> test1b\ndesc1a\ndesc1b', [('>>> test1a\n>>> test1b', 'desc1a\ndesc1b')]), ('Description\nExamples\n--------\n>>> test1\ndesc1\n>>> test2\ndesc2', [('>>> test1', 'desc1'), ('>>> test2', 'desc2')]), ('Description\nExamples\n--------\n>>> test1a\n>>> test1b\ndesc1a\ndesc1b\n>>> test2a\n>>> test2b\ndesc2a\ndesc2b\n', [('>>> test1a\n>>> test1b', 'desc1a\ndesc1b'), ('>>> test2a\n>>> test2b', 'desc2a\ndesc2b')]), ('Description\nExamples\n--------\n    >>> test1a\n    >>> test1b\n    desc1a\n    desc1b\n    >>> test2a\n    >>> test2b\n    desc2a\n    desc2b\n', [('>>> test1a\n>>> test1b', 'desc1a\ndesc1b'), ('>>> test2a\n>>> test2b', 'desc2a\ndesc2b')])])
def test_examples(source, expected_results: T.List[T.Tuple[T.Optional[str], str]]) -> None:
    """Test parsing examples."""
    docstring = parse(source)
    assert len(docstring.meta) == len(expected_results)
    for meta, expected_result in zip(docstring.meta, expected_results):
        assert meta.description == expected_result[1]
    assert len(docstring.examples) == len(expected_results)
    for example, expected_result in zip(docstring.examples, expected_results):
        assert example.snippet == expected_result[0]
        assert example.description == expected_result[1]