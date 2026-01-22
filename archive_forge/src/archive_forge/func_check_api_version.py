import logging
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
def check_api_version(check_version):
    """Validate version supplied by user

    Returns:

    * True if version is OK
    * False if the version has not been checked and the previous plugin
      check should be performed
    * throws an exception if the version is no good
    """
    from cinderclient import api_versions
    global _volume_api_version
    _volume_api_version = api_versions.get_api_version(check_version)
    if not _volume_api_version.is_latest():
        if _volume_api_version > api_versions.APIVersion('3.0'):
            if not _volume_api_version.matches(api_versions.MIN_VERSION, api_versions.MAX_VERSION):
                msg = _('versions supported by client: %(min)s - %(max)s') % {'min': api_versions.MIN_VERSION, 'max': api_versions.MAX_VERSION}
                raise exceptions.CommandError(msg)
            return True
    return False