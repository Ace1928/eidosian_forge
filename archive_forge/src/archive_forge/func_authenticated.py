from pprint import pformat
from six import iteritems
import re
@authenticated.setter
def authenticated(self, authenticated):
    """
        Sets the authenticated of this V1beta1TokenReviewStatus.
        Authenticated indicates that the token was associated with a known user.

        :param authenticated: The authenticated of this
        V1beta1TokenReviewStatus.
        :type: bool
        """
    self._authenticated = authenticated