import re
from yaql.language import specs
from yaql.language import yaqltypes
@specs.parameter('regexp', REGEX_TYPE)
@specs.parameter('string', yaqltypes.String())
@specs.name('#operator_!~')
def not_matches_operator_regex(string, regexp):
    """:yaql:operator !~

    Returns true if left doesn't match right, false otherwise.

    :signature: left !~ right
    :arg left: string to find match in
    :argType left: string
    :arg right: regex pattern
    :argType right: regex
    :returnType: boolean

    .. code::

        yaql> "acb" !~ regex("a.c")
        true
        yaql> "abc" !~ regex("a.c")
        false
    """
    return regexp.search(string) is None