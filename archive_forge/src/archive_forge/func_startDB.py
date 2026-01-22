import os
import stat
from typing import Dict, Optional
from twisted.enterprise.adbapi import (
from twisted.internet import defer, interfaces, reactor
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def startDB(self):
    import kinterbasdb
    self.DB_NAME = os.path.join(self.DB_DIR, DBTestConnector.DB_NAME)
    os.chmod(self.DB_DIR, stat.S_IRWXU + stat.S_IRWXG + stat.S_IRWXO)
    sql = 'create database "%s" user "%s" password "%s"'
    sql %= (self.DB_NAME, self.DB_USER, self.DB_PASS)
    conn = kinterbasdb.create_database(sql)
    conn.close()