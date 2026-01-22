from markdownify import markdownify as md
def test_xml_entities():
    assert md('&amp;') == '&'