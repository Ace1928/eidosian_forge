import re
from yaql.language import specs
from yaql.language import yaqltypes
@specs.parameter('regexp', REGEX_TYPE)
@specs.parameter('string', yaqltypes.String())
@specs.parameter('repl', yaqltypes.String())
@specs.parameter('count', int)
@specs.method
@specs.name('replace')
def replace_string(string, regexp, repl, count=0):
    """:yaql:replace

    Returns the string obtained by replacing the leftmost non-overlapping
    matches of regexp in string by the replacement repl, where the latter is
    only string-type.

    :signature: string.replace(regexp, repl, count => 0)
    :receiverArg string: string to make replace in
    :argType string: string
    :arg regexp: regex pattern
    :argType regexp: regex object
    :arg repl: string to replace matches of regexp
    :argType repl: string
    :arg count: how many first replaces to do. 0 by default, which means
        to do all replacements
    :argType count: integer
    :returnType: string

    .. code::

        yaql> "abcadc".replace(regex("a."), "xx")
        "xxcxxc"
    """
    return replace(regexp, string, repl, count)