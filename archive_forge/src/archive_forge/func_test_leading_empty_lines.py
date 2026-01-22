from IPython.core import inputtransformer2 as ipt2
def test_leading_empty_lines():
    for sample, expected in [LEADING_EMPTY_LINES, ONLY_EMPTY_LINES]:
        assert ipt2.leading_empty_lines(sample.splitlines(keepends=True)) == expected.splitlines(keepends=True)