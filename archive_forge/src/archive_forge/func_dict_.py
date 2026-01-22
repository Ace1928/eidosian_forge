import itertools
from yaql.language import contexts
from yaql.language import expressions
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
@specs.parameter('tuples', tuple)
@specs.inject('delegate', yaqltypes.Super(with_name=True))
@specs.no_kwargs
@specs.extension_method
def dict_(delegate, *tuples):
    """:yaql:dict

    Returns dict built from tuples.

    :signature: dict([args])
    :arg [args]: chain of tuples to be interpreted as (key, value) for dict
    :argType [args]: chain of tuples
    :returnType: dictionary

    .. code::

        yaql> dict(a => 1, b => 2)
        {"a": 1, "b": 2}
        yaql> dict(tuple(a, 1), tuple(b, 2))
        {"a": 1, "b": 2}
    """
    return delegate('dict', tuples)