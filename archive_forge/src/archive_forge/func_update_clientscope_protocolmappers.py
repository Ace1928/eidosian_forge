from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def update_clientscope_protocolmappers(self, cid, mapper_rep, realm='master'):
    """ Update an existing clientscope.

        :param cid: Id of the clientscope.
        :param mapper_rep: A ProtocolMapperRepresentation of the updated protocolmapper.
        :return HTTPResponse object on success
        """
    protocolmapper_url = URL_CLIENTSCOPE_PROTOCOLMAPPER.format(url=self.baseurl, realm=realm, id=cid, mapper_id=mapper_rep['id'])
    try:
        return open_url(protocolmapper_url, method='PUT', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, data=json.dumps(mapper_rep), validate_certs=self.validate_certs)
    except Exception as e:
        self.fail_open_url(e, msg='Could not update protocolmappers for clientscope %s in realm %s: %s' % (mapper_rep, realm, str(e)))