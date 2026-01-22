from dulwich import lru_cache
from dulwich.tests import TestCase
def size_of_list(lst):
    return sum((len(x) for x in lst))