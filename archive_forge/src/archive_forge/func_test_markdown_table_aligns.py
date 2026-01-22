import pytest
from wasabi.markdown import MarkdownRenderer
def test_markdown_table_aligns():
    md = MarkdownRenderer()
    md.add(md.table([('a', 'b', 'c')], ['foo', 'bar', 'baz'], aligns=('c', 'r', 'l')))
    expected = '| foo | bar | baz |\n| :---: | ---: | --- |\n| a | b | c |'
    assert md.text == expected
    with pytest.raises(ValueError):
        md.table([('a', 'b', 'c')], ['foo', 'bar', 'baz'], aligns=('c', 'r'))
    with pytest.raises(ValueError):
        md.table([('a', 'b', 'c')], ['foo', 'bar', 'baz'], aligns=('c', 'r', 'l', 'l'))