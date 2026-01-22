from boto.sdb.db.property import Property
from boto.sdb.db.key import Key
from boto.sdb.db.query import Query
import boto
from boto.compat import filter
@classmethod
def from_xml(cls, fp):
    xmlmanager = cls.get_xmlmanager()
    return xmlmanager.unmarshal_object(fp)