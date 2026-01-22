from __future__ import annotations
import os
import re
import shutil
import stat
import struct
import sys
import sysconfig
import warnings
from email.generator import BytesGenerator, Generator
from email.policy import EmailPolicy
from glob import iglob
from shutil import rmtree
from zipfile import ZIP_DEFLATED, ZIP_STORED
import setuptools
from setuptools import Command
from . import __version__ as wheel_version
from .macosx_libfile import calculate_macosx_platform_tag
from .metadata import pkginfo_to_metadata
from .util import log
from .vendored.packaging import tags
from .vendored.packaging import version as _packaging_version
from .wheelfile import WheelFile
def write_wheelfile(self, wheelfile_base, generator='bdist_wheel (' + wheel_version + ')'):
    from email.message import Message
    msg = Message()
    msg['Wheel-Version'] = '1.0'
    msg['Generator'] = generator
    msg['Root-Is-Purelib'] = str(self.root_is_pure).lower()
    if self.build_number is not None:
        msg['Build'] = self.build_number
    impl_tag, abi_tag, plat_tag = self.get_tag()
    for impl in impl_tag.split('.'):
        for abi in abi_tag.split('.'):
            for plat in plat_tag.split('.'):
                msg['Tag'] = '-'.join((impl, abi, plat))
    wheelfile_path = os.path.join(wheelfile_base, 'WHEEL')
    log.info(f'creating {wheelfile_path}')
    with open(wheelfile_path, 'wb') as f:
        BytesGenerator(f, maxheaderlen=0).flatten(msg)