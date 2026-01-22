import collections
import functools
import itertools
from yaql.language import exceptions
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
@specs.parameter('collection', OrderingIterable)
@specs.parameter('selector', yaqltypes.Lambda())
@specs.method
def then_by_descending(collection, selector, context):
    """:yaql:thenByDescending

    To be used with orderBy or orderByDescending. Uses selector to extract
    secondary sort key (descending) from the elements of the collection and
    adds it to the iterator.

    :signature: collection.thenByDescending(selector)
    :receiverArg collection: collection to be ordered
    :argType collection: iterable
    :arg selector: specifies a function of one argument that is used to
        extract a comparison key from each element
    :argType selector: lambda
    :returnType: iterable

    .. code::

        yaql> [[3,'c'], [2,'b'], [1,'c']].orderBy($[1]).thenByDescending($[0])
        [[2, 'b'], [3, 'c'], [1, 'c']]
    """
    collection.append_field(selector, False)
    collection.context = context
    return collection