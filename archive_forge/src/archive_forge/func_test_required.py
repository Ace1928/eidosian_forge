import logging
import time
from datetime import datetime
from boto.sdb.db.model import Model
from boto.sdb.db.property import StringProperty, IntegerProperty, BooleanProperty
from boto.sdb.db.property import DateTimeProperty, FloatProperty, ReferenceProperty
from boto.sdb.db.property import PasswordProperty, ListProperty, MapProperty
from boto.exception import SDBPersistenceError
def test_required():
    global _objects
    t = TestRequired()
    _objects['test_required_t'] = t
    t.put()
    return t