from __future__ import absolute_import, division, print_function
import platform
from ansible.module_utils import distro
from ansible.module_utils.common._utils import get_all_subclasses
def get_distribution_codename():
    """
    Return the code name for this Linux Distribution

    :rtype: NativeString or None
    :returns: A string representation of the distribution's codename or None if not a Linux distro
    """
    codename = None
    if platform.system() == 'Linux':
        os_release_info = distro.os_release_info()
        codename = os_release_info.get('version_codename')
        if codename is None:
            codename = os_release_info.get('ubuntu_codename')
        if codename is None and distro.id() == 'ubuntu':
            lsb_release_info = distro.lsb_release_info()
            codename = lsb_release_info.get('codename')
        if codename is None:
            codename = distro.codename()
            if codename == u'':
                codename = None
    return codename