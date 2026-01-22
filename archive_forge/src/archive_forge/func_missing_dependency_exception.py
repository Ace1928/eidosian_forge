from collections import defaultdict
from ..core import Store
def missing_dependency_exception(value, keyword, allowed):
    raise Exception(f'Format {value!r} does not appear to be supported.')