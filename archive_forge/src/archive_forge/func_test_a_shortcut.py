from markdownify import markdownify as md, ATX, ATX_CLOSED, BACKSLASH, UNDERSCORE
def test_a_shortcut():
    text = md('<a href="http://google.com">http://google.com</a>')
    assert text == '<http://google.com>'