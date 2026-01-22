from .exceptions import ParseException
from .util import col, replaced_by_pep8
def replace_with(repl_str):
    """
    Helper method for common parse actions that simply return
    a literal value.  Especially useful when used with
    :class:`transform_string<ParserElement.transform_string>` ().

    Example::

        num = Word(nums).set_parse_action(lambda toks: int(toks[0]))
        na = one_of("N/A NA").set_parse_action(replace_with(math.nan))
        term = na | num

        term[1, ...].parse_string("324 234 N/A 234") # -> [324, 234, nan, 234]
    """
    return lambda s, l, t: [repl_str]