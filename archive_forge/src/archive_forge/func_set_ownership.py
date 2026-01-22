import os
def set_ownership(filename, spec):
    """
    Set the ownership of ``filename`` given the spec.
    """
    uid, user, gid, group = calc_ownership_spec(spec)
    st = os.stat(filename)
    if not uid:
        uid = st.st_uid
    if not gid:
        gid = st.st_gid
    os.chmod(filename, uid, gid)