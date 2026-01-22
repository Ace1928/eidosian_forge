import logging
from email.message import Message
from email.parser import Parser
from typing import Tuple
from zipfile import BadZipFile, ZipFile
from pip._vendor.packaging.utils import canonicalize_name
from pip._internal.exceptions import UnsupportedWheel
def parse_wheel(wheel_zip: ZipFile, name: str) -> Tuple[str, Message]:
    """Extract information from the provided wheel, ensuring it meets basic
    standards.

    Returns the name of the .dist-info directory and the parsed WHEEL metadata.
    """
    try:
        info_dir = wheel_dist_info_dir(wheel_zip, name)
        metadata = wheel_metadata(wheel_zip, info_dir)
        version = wheel_version(metadata)
    except UnsupportedWheel as e:
        raise UnsupportedWheel(f'{name} has an invalid wheel, {str(e)}')
    check_compatibility(version, name)
    return (info_dir, metadata)