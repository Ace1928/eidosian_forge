from unittest import TestCase
from ipywidgets.widgets.docutils import doc_subst
def test_unused_keys(self):
    snippets = {'key': '62', 'other-key': 'unused'}

    @doc_subst(snippets)
    def f():
        """ Docstring with value {key} """
    assert f.__doc__ == ' Docstring with value 62 '