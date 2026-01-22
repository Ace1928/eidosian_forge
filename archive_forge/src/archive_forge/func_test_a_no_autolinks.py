from markdownify import markdownify as md, ATX, ATX_CLOSED, BACKSLASH, UNDERSCORE
def test_a_no_autolinks():
    assert md('<a href="https://google.com">https://google.com</a>', autolinks=False) == '[https://google.com](https://google.com)'