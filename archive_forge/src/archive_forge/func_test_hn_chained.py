from markdownify import markdownify as md, ATX, ATX_CLOSED, BACKSLASH, UNDERSCORE
def test_hn_chained():
    assert md('<h1>First</h1>\n<h2>Second</h2>\n<h3>Third</h3>', heading_style=ATX) == '# First\n\n\n## Second\n\n\n### Third\n\n'
    assert md('X<h1>First</h1>', heading_style=ATX) == 'X# First\n\n'