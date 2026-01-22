import re
from yaql.language import specs
from yaql.language import yaqltypes
@specs.parameter('string', yaqltypes.String())
@specs.parameter('regexp', yaqltypes.String())
@specs.method
def matches_(string, regexp):
    """:yaql:matches

    Returns true if string matches regexp, false otherwise.

    :signature: string.matches(regexp)
    :receiverArg string: string to find match in
    :argType string: string
    :arg regexp: regex pattern
    :argType regexp: regex object
    :returnType: boolean

    .. code::

        yaql> "abc".matches("a.c")
        true
    """
    return re.search(regexp, string) is not None