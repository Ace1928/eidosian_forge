import unittest
from wsme import WSRoot
import wsme.protocol
import wsme.rest.protocol
from wsme.root import default_prepare_response_body
from webob import Request
def test_default_transaction(self):
    import transaction
    root = WSRoot(transaction=True)
    assert root._transaction is transaction
    txn = root.begin()
    txn.abort()