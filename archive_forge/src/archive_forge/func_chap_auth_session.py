from pprint import pformat
from six import iteritems
import re
@chap_auth_session.setter
def chap_auth_session(self, chap_auth_session):
    """
        Sets the chap_auth_session of this V1ISCSIVolumeSource.
        whether support iSCSI Session CHAP authentication

        :param chap_auth_session: The chap_auth_session of this
        V1ISCSIVolumeSource.
        :type: bool
        """
    self._chap_auth_session = chap_auth_session