import stat
from base64 import standard_b64decode
from dulwich.objects import Blob, Tree
def revision_pristine_tar_data(rev):
    """Export the pristine tar data from a revision."""
    if 'deb-pristine-delta' in rev.properties:
        uuencoded = rev.properties['deb-pristine-delta']
        kind = 'gz'
    elif 'deb-pristine-delta-bz2' in rev.properties:
        uuencoded = rev.properties['deb-pristine-delta-bz2']
        kind = 'bz2'
    elif 'deb-pristine-delta-xz' in rev.properties:
        uuencoded = rev.properties['deb-pristine-delta-xz']
        kind = 'xz'
    else:
        raise KeyError(rev.revision_id)
    return (standard_b64decode(uuencoded), kind)