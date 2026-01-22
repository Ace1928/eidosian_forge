from pprint import pformat
from six import iteritems
import re
@incomplete.setter
def incomplete(self, incomplete):
    """
        Sets the incomplete of this V1beta1SubjectRulesReviewStatus.
        Incomplete is true when the rules returned by this call are incomplete.
        This is most commonly encountered when an authorizer, such as an
        external authorizer, doesn't support rules evaluation.

        :param incomplete: The incomplete of this
        V1beta1SubjectRulesReviewStatus.
        :type: bool
        """
    if incomplete is None:
        raise ValueError('Invalid value for `incomplete`, must not be `None`')
    self._incomplete = incomplete