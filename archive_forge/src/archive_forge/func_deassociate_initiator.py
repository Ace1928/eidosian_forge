from oslo_log import log as logging
import six
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils import hostutils
from os_win.utils import pathutils
from os_win.utils import win32utils
def deassociate_initiator(self, initiator, target_name):
    try:
        wt_idmethod = self._get_wt_idmethod(initiator, target_name)
        if wt_idmethod:
            wt_idmethod.Delete_()
    except exceptions.x_wmi as wmi_exc:
        err_msg = _('Could not deassociate initiator %(initiator)s from iSCSI target: %(target_name)s.')
        raise exceptions.ISCSITargetWMIException(err_msg % dict(initiator=initiator, target_name=target_name), wmi_exc=wmi_exc)