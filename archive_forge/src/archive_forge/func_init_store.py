import time
from dulwich.tests import TestCase, skipIf
from ..object_store import MemoryObjectStore
from ..objects import Blob, Commit, Tree, parse_timezone
def init_store(store, count=1):
    ret = []
    for i in range(count):
        objs = create_commit(marker=('%d' % i).encode('ascii'))
        for obj in objs:
            ret.append(obj)
            store.add_object(obj)
    return ret