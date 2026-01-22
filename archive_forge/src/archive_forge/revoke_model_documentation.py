from oslo_log import log
from oslo_serialization import msgpackutils
from oslo_utils import timeutils
from keystone.common import cache
from keystone.common import utils
See if the token matches the revocation event.

    A brute force approach to checking.
    Compare each attribute from the event with the corresponding
    value from the token.  If the event does not have a value for
    the attribute, a match is still possible.  If the event has a
    value for the attribute, and it does not match the token, no match
    is possible, so skip the remaining checks.

    :param event: a RevokeEvent instance
    :param token_values: dictionary with set of values taken from the
                         token
    :returns: True if the token matches the revocation event, indicating the
              token has been revoked
    