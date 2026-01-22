from .ast import AtRule, Declaration, ParseError, QualifiedRule
from .tokenizer import parse_component_value_list
def parse_one_component_value(input, skip_comments=False):
    """Parse a single :diagram:`component value`.

    This is used e.g. for an attribute value
    referred to by ``attr(foo length)``.

    :type input: :obj:`str` or :term:`iterable`
    :param input: A string or an iterable of :term:`component values`.
    :type skip_comments: :obj:`bool`
    :param skip_comments: If the input is a string, ignore all CSS comments.
    :returns:
        A :term:`component value` (that is neither whitespace or comment),
        or a :class:`~tinycss2.ast.ParseError`.

    """
    tokens = _to_token_iterator(input, skip_comments)
    first = _next_significant(tokens)
    second = _next_significant(tokens)
    if first is None:
        return ParseError(1, 1, 'empty', 'Input is empty')
    if second is not None:
        return ParseError(second.source_line, second.source_column, 'extra-input', 'Got more than one token')
    else:
        return first