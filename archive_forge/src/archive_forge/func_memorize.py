import collections
import functools
import itertools
from yaql.language import exceptions
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
@specs.parameter('collection', yaqltypes.Iterable())
@specs.method
def memorize(collection, engine):
    """:yaql:memorize

    Returns an iterator over collection and memorizes already iterated values.
    This function can be used for iterating over collection several times
    as it remembers elements, and when given collection (iterator) is too
    large to be unwrapped at once.

    :signature: collection.memorize()
    :receiverArg collection: input collection
    :argType collection: iterable
    :returnType: iterator to collection

    .. code::

        yaql> let(range(4)) -> $.sum() + $.len()
        6
        yaql> let(range(4).memorize()) -> $.sum() + $.len()
        10
    """
    return utils.memorize(collection, engine)