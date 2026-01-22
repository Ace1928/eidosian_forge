import collections
import re
import sys
from yaql.language import exceptions
from yaql.language import lexer
def get_max_collection_size(engine):
    return engine.options.get('yaql.limitIterators', -1)