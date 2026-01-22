from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def get_realm_info_by_id(self, realm='master'):
    """ Obtain realm public info by id

        :param realm: realm id
        :return: dict of real, representation or None if none matching exist
        """
    realm_info_url = URL_REALM_INFO.format(url=self.baseurl, realm=realm)
    try:
        return json.loads(to_native(open_url(realm_info_url, method='GET', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, validate_certs=self.validate_certs).read()))
    except HTTPError as e:
        if e.code == 404:
            return None
        else:
            self.fail_open_url(e, msg='Could not obtain realm %s: %s' % (realm, str(e)), exception=traceback.format_exc())
    except ValueError as e:
        self.module.fail_json(msg='API returned incorrect JSON when trying to obtain realm %s: %s' % (realm, str(e)), exception=traceback.format_exc())
    except Exception as e:
        self.module.fail_json(msg='Could not obtain realm %s: %s' % (realm, str(e)), exception=traceback.format_exc())