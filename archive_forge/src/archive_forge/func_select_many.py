import collections
import functools
import itertools
from yaql.language import exceptions
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
@specs.parameter('collection', yaqltypes.Iterable())
@specs.parameter('selector', yaqltypes.Lambda())
@specs.method
def select_many(collection, selector):
    """:yaql:selectMany

    Applies a selector to each element of the collection and returns an
    iterator over results. If the selector returns an iterable object,
    iterates over its elements instead of itself.

    :signature: collection.selectMany(selector)
    :receiverArg collection: input collection
    :argType collection: iterable
    :arg selector: function to be applied to every collection element
    :argType selector: lambda
    :returnType: iterator

    .. code::

        yaql> [0, 1, 2].selectMany($ + 2)
        [2, 3, 4]
        yaql> [0, [1, 2], 3].selectMany($ * 2)
        [0, 1, 2, 1, 2, 6]
    """
    for item in collection:
        inner = selector(item)
        if utils.is_iterable(inner):
            for t in inner:
                yield t
        else:
            yield inner