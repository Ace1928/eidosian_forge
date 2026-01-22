import torch._dynamo.test_case
import unittest.mock
import os
import contextlib
import torch._logging
import torch._logging._internal
from torch._dynamo.utils import LazyString
import logging
def getRecord(self, records, m):
    record = None
    for r in records:
        if m in r.getMessage():
            self.assertIsNone(record, msg=LazyString(lambda: f'multiple matching records: {record} and {r} among {records}'))
            record = r
    if record is None:
        self.fail(f'did not find record with {m} among {records}')
    return record