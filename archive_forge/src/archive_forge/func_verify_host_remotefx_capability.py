import socket
from oslo_log import log as logging
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils.winapi import libs as w_lib
def verify_host_remotefx_capability(self):
    """Validates that the host supports RemoteFX.

        :raises exceptions.HyperVRemoteFXException: if the host has no GPU
            that supports DirectX 11, or SLAT.
        """
    synth_3d_video_pool = self._conn.Msvm_Synth3dVideoPool()[0]
    if not synth_3d_video_pool.IsGpuCapable:
        raise exceptions.HyperVRemoteFXException(_('To enable RemoteFX on Hyper-V at least one GPU supporting DirectX 11 is required.'))
    if not synth_3d_video_pool.IsSlatCapable:
        raise exceptions.HyperVRemoteFXException(_('To enable RemoteFX on Hyper-V it is required that the host GPUs support SLAT.'))