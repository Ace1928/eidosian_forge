import os
import stat
from typing import Dict, Optional
from twisted.enterprise.adbapi import (
from twisted.internet import defer, interfaces, reactor
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def withConnection(self, conn):
    curs = conn.cursor()
    try:
        curs.execute('select x from simple order by x')
        for i in range(self.num_iterations):
            row = curs.fetchone()
            self.assertTrue(len(row) == 1, 'Wrong size row')
            self.assertTrue(row[0] == i, 'Value not returned.')
    finally:
        curs.close()
    return 'done'