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
def pbm_wsdl_loc_set(self, pbm_wsdl_loc):
    self._pbm_wsdl_loc = pbm_wsdl_loc
    self._pbm = None
    LOG.info('PBM WSDL updated to %s', pbm_wsdl_loc)