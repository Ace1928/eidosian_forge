from markdownify import markdownify as md
def test_ignore_comments():
    text = md('<!-- This is a comment -->')
    assert text == ''