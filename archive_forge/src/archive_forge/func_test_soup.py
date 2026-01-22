from markdownify import markdownify as md
def test_soup():
    assert md('<div><span>Hello</div></span>') == 'Hello'