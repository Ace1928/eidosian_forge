import builtins
import ctypes.wintypes
from paramiko.util import u
def get_security_attributes_for_user(user=None):
    """
    Return a SECURITY_ATTRIBUTES structure with the SID set to the
    specified user (uses current user if none is specified).
    """
    if user is None:
        user = get_current_user()
    assert isinstance(user, TOKEN_USER), 'user must be TOKEN_USER instance'
    SD = SECURITY_DESCRIPTOR()
    SA = SECURITY_ATTRIBUTES()
    SA.descriptor = SD
    SA.bInheritHandle = 1
    ctypes.windll.advapi32.InitializeSecurityDescriptor(ctypes.byref(SD), SECURITY_DESCRIPTOR.REVISION)
    ctypes.windll.advapi32.SetSecurityDescriptorOwner(ctypes.byref(SD), user.SID, 0)
    return SA