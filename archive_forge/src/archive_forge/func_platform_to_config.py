from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import json
import os
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
def platform_to_config(platform):
    if not platform:
        platform = platforms.Platform.Current()
    if platform.operating_system == platforms.OperatingSystem.MACOSX:
        return ConfigType.KEYCHAIN
    elif platform.operating_system == platforms.OperatingSystem.LINUX:
        return ConfigType.PKCS11
    elif platform.operating_system == platforms.OperatingSystem.WINDOWS:
        return ConfigType.MYSTORE
    else:
        raise ECPConfigError('Unsupported platform {}. Enterprise Certificate Proxy currently only supports OSX, Windows, and Linux.'.format(platform.operating_system))