import itertools
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
import yaql.standard_library.queries
@specs.method
@specs.name('deleteAll')
@specs.parameter('d', utils.MappingType, alias='dict')
@specs.parameter('keys', yaqltypes.Iterable())
def delete_keys_seq(d, keys):
    """:yaql:deleteAll

    Returns dict with keys removed. Keys are provided as an iterable
    collection.

    :signature: dict.deleteAll(keys)
    :receiverArg dict: input dictionary
    :argType dict: mapping
    :arg keys: keys to be removed from dictionary
    :argType keys: iterable
    :returnType: mapping

    .. code::

        yaql> {"a" => 1, "b" => 2, "c" => 3}.deleteAll(["a", "c"])
        {"b": 2}
    """
    copy = dict(d)
    for t in keys:
        copy.pop(t, None)
    return copy