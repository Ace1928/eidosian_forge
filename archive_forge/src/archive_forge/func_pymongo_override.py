import collections.abc
import copy
import functools
from dogpile.cache import region as dp_region
from oslo_utils import uuidutils
from oslo_cache.backends import mongo
from oslo_cache import core
from oslo_cache import exception
from oslo_cache.tests import test_cache
def pymongo_override():
    global pymongo
    import pymongo
    if pymongo.MongoClient is not MockMongoClient:
        pymongo.MongoClient = MockMongoClient