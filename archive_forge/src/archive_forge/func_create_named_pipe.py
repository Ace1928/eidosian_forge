import sys
def create_named_pipe(pipename, openMode=None, pipeMode=None, nMaxInstances=None, nOutBufferSize=None, nInBufferSize=None, nDefaultTimeOut=None, saAttr=-1):
    if openMode is None:
        openMode = win32con.PIPE_ACCESS_DUPLEX | win32con.FILE_FLAG_OVERLAPPED
    if pipeMode is None:
        pipeMode = win32con.PIPE_TYPE_MESSAGE | win32con.PIPE_READMODE_BYTE | win32con.PIPE_WAIT
    if nMaxInstances is None:
        nMaxInstances = 64
    if nOutBufferSize is None:
        nOutBufferSize = 65000
    if nInBufferSize is None:
        nInBufferSize = 65000
    if nDefaultTimeOut is None:
        nDefaultTimeOut = 0
    if saAttr == -1:
        saAttr = win32security.SECURITY_ATTRIBUTES()
        sia = ntsecuritycon.SECURITY_NT_AUTHORITY
        remoteAccessSid = win32security.SID()
        remoteAccessSid.Initialize(sia, 1)
        remoteAccessSid.SetSubAuthority(0, ntsecuritycon.SECURITY_NETWORK_RID)
        allowedPsids = []
        allowedPsid_0 = win32security.SID()
        allowedPsid_0.Initialize(sia, 1)
        allowedPsid_0.SetSubAuthority(0, ntsecuritycon.SECURITY_LOCAL_SYSTEM_RID)
        allowedPsid_1 = win32security.SID()
        allowedPsid_1.Initialize(sia, 2)
        allowedPsid_1.SetSubAuthority(0, ntsecuritycon.SECURITY_BUILTIN_DOMAIN_RID)
        allowedPsid_1.SetSubAuthority(1, ntsecuritycon.DOMAIN_ALIAS_RID_ADMINS)
        allowedPsids.append(allowedPsid_0)
        allowedPsids.append(allowedPsid_1)
        acl = win32security.ACL()
        acl.Initialize()
        acl.AddAccessDeniedAce(win32security.ACL_REVISION, ntsecuritycon.GENERIC_ALL, remoteAccessSid)
        for allowedPsid in allowedPsids:
            acl.AddAccessAllowedAce(win32security.ACL_REVISION, ntsecuritycon.GENERIC_ALL, allowedPsid)
        sd = win32security.SECURITY_DESCRIPTOR()
        sd.Initialize()
        sd.SetSecurityDescriptorDacl(True, acl, False)
        saAttr.bInheritHandle = 1
        saAttr.SECURITY_DESCRIPTOR = sd
    try:
        npipe = win32pipe.CreateNamedPipe(pipename, openMode, pipeMode, nMaxInstances, nOutBufferSize, nInBufferSize, nDefaultTimeOut, saAttr)
        if npipe == win32file.INVALID_HANDLE_VALUE:
            return None
        return npipe
    except pywintypes.error:
        return None