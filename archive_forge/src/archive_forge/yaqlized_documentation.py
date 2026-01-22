import re
from yaql.language import expressions
from yaql.language import runner
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
from yaql import yaqlization
:yaql:operator indexer

    Returns value of attribute/property key of the object.

    :signature: obj[key]
    :arg obj: yaqlized object
    :argType obj: yaqlized object, initialized with
        yaqlize_indexer equal to True
    :arg key: index name
    :argType key: keyword
    :returnType: any
    