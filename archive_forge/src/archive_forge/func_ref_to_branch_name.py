from dulwich.objects import Tag, object_class
from dulwich.refs import (LOCAL_BRANCH_PREFIX, LOCAL_TAG_PREFIX)
from dulwich.repo import RefsContainer
from .. import controldir, errors, osutils
from .. import revision as _mod_revision
def ref_to_branch_name(ref):
    """Map a ref to a branch name

    :param ref: Ref
    :return: A branch name
    """
    if ref == b'HEAD':
        return ''
    if ref is None:
        return ref
    if ref.startswith(LOCAL_BRANCH_PREFIX):
        return ref[len(LOCAL_BRANCH_PREFIX):].decode('utf-8')
    raise ValueError('unable to map ref %s back to branch name' % ref)