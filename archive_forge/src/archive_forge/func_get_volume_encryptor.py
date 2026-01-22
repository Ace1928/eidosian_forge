from __future__ import annotations
from typing import Any
from oslo_log import log as logging
from oslo_utils import importutils
from oslo_utils import strutils
from os_brick.encryptors import base
from os_brick.encryptors import nop
def get_volume_encryptor(root_helper: str, connection_info: dict[str, Any], keymgr, execute=None, *args, **kwargs) -> base.VolumeEncryptor:
    """Creates a VolumeEncryptor used to encrypt the specified volume.

    :param: the connection information used to attach the volume
    :returns VolumeEncryptor: the VolumeEncryptor for the volume
    """
    encryptor = nop.NoOpEncryptor(*args, root_helper=root_helper, connection_info=connection_info, keymgr=keymgr, execute=execute, **kwargs)
    location = kwargs.get('control_location', None)
    if location and location.lower() == 'front-end':
        provider = kwargs.get('provider')
        if provider in LEGACY_PROVIDER_CLASS_TO_FORMAT_MAP:
            LOG.warning('Use of the in tree encryptor class %(provider)s by directly referencing the implementation class will be blocked in the Queens release of os-brick.', {'provider': provider})
            provider = LEGACY_PROVIDER_CLASS_TO_FORMAT_MAP[provider]
        if provider in FORMAT_TO_FRONTEND_ENCRYPTOR_MAP:
            provider = FORMAT_TO_FRONTEND_ENCRYPTOR_MAP[provider]
        elif provider is None:
            provider = 'os_brick.encryptors.nop.NoOpEncryptor'
        else:
            LOG.warning('Use of the out of tree encryptor class %(provider)s will be blocked with the Queens release of os-brick.', {'provider': provider})
        try:
            encryptor = importutils.import_object(provider, root_helper, connection_info, keymgr, execute, **kwargs)
        except Exception as e:
            LOG.error('Error instantiating %(provider)s: %(exception)s', {'provider': provider, 'exception': e})
            raise
    msg = "Using volume encryptor '%(encryptor)s' for connection: %(connection_info)s" % {'encryptor': encryptor, 'connection_info': connection_info}
    LOG.debug(strutils.mask_password(msg))
    return encryptor