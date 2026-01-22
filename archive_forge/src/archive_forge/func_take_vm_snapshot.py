import functools
import time
import uuid
from eventlet import patcher
from eventlet import tpool
from oslo_log import log as logging
from oslo_utils import uuidutils
from six.moves import range  # noqa
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
from os_win.utils import jobutils
from os_win.utils import pathutils
def take_vm_snapshot(self, vm_name, snapshot_name=None):
    vm = self._lookup_vm_check(vm_name, as_vssd=False)
    vs_snap_svc = self._compat_conn.Msvm_VirtualSystemSnapshotService()[0]
    job_path, snp_setting_data, ret_val = vs_snap_svc.CreateSnapshot(AffectedSystem=vm.path_(), SnapshotType=self._SNAPSHOT_FULL)
    job = self._jobutils.check_ret_val(ret_val, job_path)
    snp_setting_data = job.associators(wmi_result_class=self._VIRTUAL_SYSTEM_SETTING_DATA_CLASS, wmi_association_class=self._AFFECTED_JOB_ELEMENT_CLASS)[0]
    if snapshot_name is not None:
        snp_setting_data.ElementName = snapshot_name
        self._modify_virtual_system(snp_setting_data)
    return snp_setting_data.path_()