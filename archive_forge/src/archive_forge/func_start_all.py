import boto
from boto.sdb.db.property import StringProperty, DateTimeProperty, IntegerProperty
from boto.sdb.db.model import Model
import datetime, subprocess, time
from boto.compat import StringIO
@classmethod
def start_all(cls, queue_name):
    for task in cls.all():
        task.start(queue_name)