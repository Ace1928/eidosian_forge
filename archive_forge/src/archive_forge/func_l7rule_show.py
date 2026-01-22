import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def l7rule_show(self, l7rule_id, l7policy_id):
    """Show a l7rule's settings

        :param string l7rule_id:
            ID of the l7rule to show
        :param string l7policy_id:
            ID of the l7policy for this l7rule
        :return:
            Dict of the specified l7rule's settings
        """
    url = const.BASE_L7RULE_URL.format(policy_uuid=l7policy_id)
    response = self._find(path=url, value=l7rule_id)
    return response