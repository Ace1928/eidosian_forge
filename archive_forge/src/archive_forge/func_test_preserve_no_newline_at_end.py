from .roundtrip import dedent
def test_preserve_no_newline_at_end(self):
    x = dedent('\n        123')
    assert x == '123'