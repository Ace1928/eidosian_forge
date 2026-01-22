from markdownify import markdownify as md
def test_special_tags():
    assert md('<!DOCTYPE html>') == ''
    assert md('<![CDATA[foobar]]>') == 'foobar'