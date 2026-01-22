from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def delete_clientscope(self, name=None, cid=None, realm='master'):
    """ Delete a clientscope. One of name or cid must be provided.

        Providing the clientscope ID is preferred as it avoids a second lookup to
        convert a clientscope name to an ID.

        :param name: The name of the clientscope. A lookup will be performed to retrieve the clientscope ID.
        :param cid: The ID of the clientscope (preferred to name).
        :param realm: The realm in which this group resides, default "master".
        """
    if cid is None and name is None:
        raise Exception('Unable to delete group - one of group ID or name must be provided.')
    if cid is None and name is not None:
        for clientscope in self.get_clientscopes(realm=realm):
            if clientscope['name'] == name:
                cid = clientscope['id']
                break
    if cid is None:
        return None
    clientscope_url = URL_CLIENTSCOPE.format(realm=realm, id=cid, url=self.baseurl)
    try:
        return open_url(clientscope_url, method='DELETE', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, validate_certs=self.validate_certs)
    except Exception as e:
        self.fail_open_url(e, msg='Unable to delete clientscope %s: %s' % (cid, str(e)))