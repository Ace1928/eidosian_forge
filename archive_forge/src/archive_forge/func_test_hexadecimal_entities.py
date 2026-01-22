from markdownify import markdownify as md
def test_hexadecimal_entities():
    assert md('&#x27;') == "'"