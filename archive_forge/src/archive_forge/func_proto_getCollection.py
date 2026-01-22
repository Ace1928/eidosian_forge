import os
from twisted.spread import pb
def proto_getCollection(self, requestID, name, domain, password):
    collection = self._getCollection()
    if collection is None:
        self.sendError(requestID, 'permission denied')
    else:
        self.sendAnswer(requestID, collection)