import re
from yaql.language import specs
from yaql.language import yaqltypes
@specs.parameter('regexp', REGEX_TYPE)
@specs.parameter('string', yaqltypes.String())
@specs.parameter('repl', yaqltypes.Lambda(with_context=True))
@specs.parameter('count', int)
@specs.method
@specs.name('replaceBy')
def replace_by_string(context, string, regexp, repl, count=0):
    """:yaql:replaceBy

    Replaces matches of regexp in string with values provided by the
    supplied function.

    :signature: string.replaceBy(regexp, repl, count => 0)
    :receiverArg string: string to make replace in
    :argType string: string
    :arg regexp: regex pattern
    :argType regexp: regex object
    :arg repl: lambda function which returns string to make replacements
        according to input matches
    :argType repl: lambda
    :arg count: how many first replaces to do. 0 by default, which means
        to do all replacements
    :argType count: integer
    :returnType: string

    .. code::

        yaql> "abcadc".replaceBy(regex("a.c"), switch($.value = "abc" => xx,
                                                      $.value = "adc" => yy))
        "xxyy"
    """
    return replace_by(context, regexp, string, repl, count)