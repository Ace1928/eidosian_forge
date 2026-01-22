import abc
from keystone import exception
@abc.abstractmethod
def list_trusts(self):
    raise exception.NotImplemented()