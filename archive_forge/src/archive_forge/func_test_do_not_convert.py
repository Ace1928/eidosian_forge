from markdownify import markdownify as md
def test_do_not_convert():
    text = md('<a href="https://github.com/matthewwithanm">Some Text</a>', convert=[])
    assert text == 'Some Text'