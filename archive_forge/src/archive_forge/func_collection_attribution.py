import collections
import functools
import itertools
from yaql.language import exceptions
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
@specs.parameter('collection', yaqltypes.Iterable())
@specs.parameter('attribute', yaqltypes.Keyword(expand=False))
@specs.inject('operator', yaqltypes.Delegate('#operator_.'))
@specs.name('#operator_.')
def collection_attribution(collection, attribute, operator):
    """:yaql:operator .

    Retrieves the value of an attribute for each element in a collection and
    returns a list of results.

    :signature: collection.attribute
    :arg collection: input collection
    :argType collection: iterable
    :arg attribute: attribute to get on every collection item
    :argType attribute: keyword
    :returnType: list

    .. code::

        yaql> [{"a" => 1}, {"a" => 2, "b" => 3}].a
        [1, 2]
    """
    return map(lambda t: operator(t, attribute), collection)