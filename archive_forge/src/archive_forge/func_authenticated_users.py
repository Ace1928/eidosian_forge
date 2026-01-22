import collections
import collections.abc
import operator
import warnings
@staticmethod
def authenticated_users():
    """Factory method for a member representing all authenticated users.

        Returns:
            str: A member string representing all authenticated users.
        """
    return 'allAuthenticatedUsers'