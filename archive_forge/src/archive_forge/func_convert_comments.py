import re
def convert_comments(text):
    """preprocess old style comments.

    example:

    from mako.ext.preprocessors import convert_comments
    t = Template(..., preprocessor=convert_comments)"""
    return re.sub('(?<=\\n)\\s*#[^#]', '##', text)