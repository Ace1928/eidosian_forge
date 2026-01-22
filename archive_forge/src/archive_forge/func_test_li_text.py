from markdownify import markdownify as md
def test_li_text():
    assert md('<ul><li>foo <a href="#">bar</a></li><li>foo bar  </li><li>foo <b>bar</b>   <i>space</i>.</ul>') == '* foo [bar](#)\n* foo bar\n* foo **bar** *space*.\n'