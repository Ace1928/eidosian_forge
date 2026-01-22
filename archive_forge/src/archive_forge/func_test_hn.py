from markdownify import markdownify as md, ATX, ATX_CLOSED, BACKSLASH, UNDERSCORE
def test_hn():
    assert md('<h3>Hello</h3>') == '### Hello\n\n'
    assert md('<h4>Hello</h4>') == '#### Hello\n\n'
    assert md('<h5>Hello</h5>') == '##### Hello\n\n'
    assert md('<h6>Hello</h6>') == '###### Hello\n\n'