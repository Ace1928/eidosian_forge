import collections
import re
import sys
from yaql.language import exceptions
from yaql.language import lexer
def get_memory_quota(engine):
    return engine.options.get('yaql.memoryQuota', -1)