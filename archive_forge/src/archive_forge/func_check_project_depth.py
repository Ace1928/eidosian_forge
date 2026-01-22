import abc
import keystone.conf
from keystone import exception
def check_project_depth(self, max_depth):
    """Check the projects depth in the backend whether exceed the limit.

        :param max_depth: the limit depth that project depth should not exceed.
        :type max_depth: integer

        :returns: the exceeded project's id or None if no exceeding.

        """
    raise exception.NotImplemented()