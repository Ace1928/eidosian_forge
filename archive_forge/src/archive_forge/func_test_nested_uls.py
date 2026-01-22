from markdownify import markdownify as md
def test_nested_uls():
    """
    Nested ULs should alternate bullet characters.

    """
    assert md(nested_uls) == '\n* 1\n\t+ a\n\t\t- I\n\t\t- II\n\t\t- III\n\t+ b\n\t+ c\n* 2\n* 3\n'