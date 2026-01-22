from textwrap import dedent
from pandas.util._decorators import doc
def test_inherit_doc_template():
    docstr = dedent('\n        This is the cummin method.\n\n        It computes the cumulative minimum.\n        ')
    assert cummin.__doc__ == docstr