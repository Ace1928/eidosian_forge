from markdownify import markdownify as md
def test_nested_ols():
    assert md(nested_ols) == '\n1. 1\n\t1. a\n\t\t1. I\n\t\t2. II\n\t\t3. III\n\t2. b\n\t3. c\n2. 2\n3. 3\n'