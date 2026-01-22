from markdownify import markdownify as md, ATX, ATX_CLOSED, BACKSLASH, UNDERSCORE
def test_hn_nested_tag_heading_style():
    assert md('<h1>A <p>P</p> C </h1>', heading_style=ATX_CLOSED) == '# A P C #\n\n'
    assert md('<h1>A <p>P</p> C </h1>', heading_style=ATX) == '# A P C\n\n'