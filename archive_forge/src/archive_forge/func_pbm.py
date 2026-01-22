import logging
from oslo_concurrency import lockutils
from oslo_context import context
from oslo_utils import excutils
from oslo_utils import reflection
from oslo_vmware._i18n import _
from oslo_vmware.common import loopingcall
from oslo_vmware import exceptions
from oslo_vmware import pbm
from oslo_vmware import vim
from oslo_vmware import vim_util
@property
def pbm(self):
    if not self._pbm and self._pbm_wsdl_loc:
        self._pbm = pbm.Pbm(protocol=self._scheme, host=self._host, port=self._port, wsdl_url=self._pbm_wsdl_loc, cacert=self._cacert, insecure=self._insecure, pool_maxsize=self._pool_size, connection_timeout=self._connection_timeout, op_id_prefix=self._op_id_prefix)
        if self._session_id:
            self._pbm.set_soap_cookie(self._vim.get_http_cookie())
    return self._pbm