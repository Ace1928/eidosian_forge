from markdownify import markdownify as md
def test_named_entities():
    assert md('&raquo;') == u'»'