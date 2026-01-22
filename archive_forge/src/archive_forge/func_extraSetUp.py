import os
import stat
from typing import Dict, Optional
from twisted.enterprise.adbapi import (
from twisted.internet import defer, interfaces, reactor
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def extraSetUp(self):
    """
        Skip the test if C{good_sql} is unavailable.  Otherwise, set up the
        database, create a connection pool pointed at it, and set up a simple
        schema in it.
        """
    if self.good_sql is None:
        raise unittest.SkipTest('no good sql for reconnect test')
    self.startDB()
    self.dbpool = self.makePool(cp_max=1, cp_reconnect=True, cp_good_sql=self.good_sql)
    self.dbpool.start()
    return self.dbpool.runOperation(simple_table_schema)