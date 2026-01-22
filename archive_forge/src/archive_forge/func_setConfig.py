import boto
import boto.utils
from boto.compat import StringIO
from boto.mashups.iobject import IObject
from boto.pyami.config import Config, BotoConfigPath
from boto.mashups.interactive import interactive_shell
from boto.sdb.db.model import Model
from boto.sdb.db.property import StringProperty
import os
def setConfig(self, config):
    local_file = '%s.ini' % self.instance.id
    fp = open(local_file)
    config.write(fp)
    fp.close()
    self.put_file(local_file, BotoConfigPath)
    self._config = config