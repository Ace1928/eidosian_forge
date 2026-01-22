import collections
import functools
import itertools
from yaql.language import exceptions
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
:yaql:groupBy

        Returns a collection grouped by keySelector with applied valueSelector
        as values. Returns a list of pairs where the first value is a result
        value of keySelector and the second is a list of values which have
        common keySelector return value.

        :signature: collection.groupBy(keySelector, valueSelector => null,
                                       aggregator => null)
        :receiverArg collection: input collection
        :argType collection: iterable
        :arg keySelector: function to be applied to every collection element.
            Values are grouped by return value of this function
        :argType keySelector: lambda
        :arg valueSelector: function to be applied to every collection element
            to put it under appropriate group. null by default, which means
            return element itself
        :argType valueSelector: lambda
        :arg aggregator: function to aggregate value within each group. null by
            default, which means no function to be evaluated on groups
        :argType aggregator: lambda
        :returnType: list

        .. code::

            yaql> [["a", 1], ["b", 2], ["c", 1], ["d", 2]].groupBy($[1], $[0])
            [[1, ["a", "c"]], [2, ["b", "d"]]]
            yaql> [["a", 1], ["b", 2], ["c", 1]].groupBy($[1], $[0], $.sum())
            [[1, "ac"], [2, "b"]]
        