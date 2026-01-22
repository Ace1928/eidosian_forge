import collections
import collections.abc
import operator
import warnings
@staticmethod
def service_account(email):
    """Factory method for a service account member.

        Args:
            email (str): E-mail for this particular service account.

        Returns:
            str: A member string corresponding to the given service account.

        """
    return 'serviceAccount:%s' % (email,)