import sys
import os
from numpy.core import dtype
from numpy.core import numerictypes as _numerictypes
from numpy.core.function_base import add_newdoc
def numeric_type_aliases(aliases):

    def type_aliases_gen():
        for alias, doc in aliases:
            try:
                alias_type = getattr(_numerictypes, alias)
            except AttributeError:
                pass
            else:
                yield (alias_type, alias, doc)
    return list(type_aliases_gen())