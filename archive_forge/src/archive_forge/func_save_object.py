import boto
import re
from boto.utils import find_class
import uuid
from boto.sdb.db.key import Key
from boto.sdb.db.blob import Blob
from boto.sdb.db.property import ListProperty, MapProperty
from datetime import datetime, date, time
from boto.exception import SDBPersistenceError, S3ResponseError
from boto.compat import map, six, long_type
def save_object(self, obj, expected_value=None):
    if not obj.id:
        obj.id = str(uuid.uuid4())
    attrs = {'__type__': obj.__class__.__name__, '__module__': obj.__class__.__module__, '__lineage__': obj.get_lineage()}
    del_attrs = []
    for property in obj.properties(hidden=False):
        value = property.get_value_for_datastore(obj)
        if value is not None:
            value = self.encode_value(property, value)
        if value == []:
            value = None
        if value is None:
            del_attrs.append(property.name)
            continue
        attrs[property.name] = value
        if property.unique:
            try:
                args = {property.name: value}
                obj2 = next(obj.find(**args))
                if obj2.id != obj.id:
                    raise SDBPersistenceError('Error: %s must be unique!' % property.name)
            except StopIteration:
                pass
    if expected_value:
        prop = obj.find_property(expected_value[0])
        v = expected_value[1]
        if v is not None and (not isinstance(v, bool)):
            v = self.encode_value(prop, v)
        expected_value[1] = v
    self.domain.put_attributes(obj.id, attrs, replace=True, expected_value=expected_value)
    if len(del_attrs) > 0:
        self.domain.delete_attributes(obj.id, del_attrs)
    return obj