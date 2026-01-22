import logging
from oslo_utils import timeutils
from suds import sudsobject
def get_vc_version(session):
    """Return the dot-separated vCenter version string. For example, "1.2".

    :param session: vCenter soap session
    :return: vCenter version
    """
    return session.vim.service_content.about.version