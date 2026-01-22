from dulwich.objects import Tag, object_class
from dulwich.refs import (LOCAL_BRANCH_PREFIX, LOCAL_TAG_PREFIX)
from dulwich.repo import RefsContainer
from .. import controldir, errors, osutils
from .. import revision as _mod_revision
def tag_name_to_ref(name):
    """Map a tag name to a ref.

    :param name: Tag name
    :return: ref string
    """
    return LOCAL_TAG_PREFIX + osutils.safe_utf8(name)