import string as string_module
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
@specs.parameter('string', yaqltypes.String())
@specs.parameter('replacements', utils.MappingType)
@specs.parameter('count', int)
@specs.inject('str_func', yaqltypes.Delegate('str'))
@specs.method
@specs.name('replace')
def replace_with_dict(string, str_func, replacements, count=-1):
    """:yaql:replace

    Returns a string with all occurrences of replacements' keys replaced
    with corresponding replacements' values.
    If count is specified, only the first count occurrences of every key
    are replaced.

    :signature: string.replace(replacements, count => -1)
    :receiverArg string: input string
    :argType string: string
    :arg replacements: dict of replacements in format {old => new ...}
    :argType replacements: mapping
    :arg count: how many first occurrences of every key are replaced. -1 by
        default, which means to do all replacements
    :argType count: integer
    :returnType: string

    .. code::

        yaql> "abc ab abc".replace({abc => xx, ab => yy})
        "xx yy xx"
        yaql> "abc ab abc".replace({ab => yy, abc => xx})
        "yyc yy yyc"
        yaql> "abc ab abc".replace({ab => yy, abc => xx}, 1)
        "yyc ab xx"
    """
    for key, value in replacements.items():
        string = string.replace(str_func(key), str_func(value), count)
    return string