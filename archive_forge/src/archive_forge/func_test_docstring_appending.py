from textwrap import dedent
from pandas.util._decorators import doc
def test_docstring_appending():
    docstr = dedent('\n        This is the cumavg method.\n\n        It computes the cumulative average.\n\n        Examples\n        --------\n\n        >>> cumavg([1, 2, 3])\n        2\n        ')
    assert cumavg.__doc__ == docstr