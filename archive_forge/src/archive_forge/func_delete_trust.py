import abc
from keystone import exception
@abc.abstractmethod
def delete_trust(self, trust_id):
    raise exception.NotImplemented()