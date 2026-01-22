from collections.abc import Mapping, Hashable
import collections
import copy
from ..language import ast
from ..pyutils.cached_property import cached_property
from ..pyutils.ordereddict import OrderedDict
from ..utils.assert_valid_name import assert_valid_name
def is_same_type(self, other):
    return isinstance(other, GraphQLNonNull) and self.of_type.is_same_type(other.of_type)