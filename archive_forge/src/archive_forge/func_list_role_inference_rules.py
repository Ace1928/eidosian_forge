import abc
import keystone.conf
from keystone import exception
@abc.abstractmethod
def list_role_inference_rules(self):
    """List all the rules used to imply one role from another."""
    raise exception.NotImplemented()