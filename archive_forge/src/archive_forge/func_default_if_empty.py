import collections
import functools
import itertools
from yaql.language import exceptions
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
@specs.method
@specs.parameter('collection', yaqltypes.Iterable())
@specs.parameter('default', yaqltypes.Iterable())
def default_if_empty(engine, collection, default):
    """:yaql:defaultIfEmpty

    Returns default value if collection is empty.

    :signature: collection.defaultIfEmpty(default)
    :receiverArg collection: input collection
    :argType collection: iterable
    :arg default: value to be returned if collection size is 0
    :argType default: iterable
    :returnType: iterable

    .. code::

        yaql> [].defaultIfEmpty([1, 2])
        [1, 2]
    """
    if isinstance(collection, (utils.SequenceType, utils.SetType)):
        return default if len(collection) == 0 else collection
    collection = memorize(collection, engine)
    it = iter(collection)
    try:
        next(it)
        return collection
    except StopIteration:
        return default