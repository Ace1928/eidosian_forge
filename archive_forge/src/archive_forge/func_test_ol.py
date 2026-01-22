from markdownify import markdownify as md
def test_ol():
    assert md('<ol><li>a</li><li>b</li></ol>') == '1. a\n2. b\n'
    assert md('<ol start="3"><li>a</li><li>b</li></ol>') == '3. a\n4. b\n'