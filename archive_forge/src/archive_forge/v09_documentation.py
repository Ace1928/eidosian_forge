from ...testament import StrictTestament3
from ..bundle_data import BundleInfo
from . import _get_bundle_header
from .v08 import BundleReader, BundleSerializerV08
Read the rest of the bundles from the supplied file.

        :param f: The file to read from
        :return: A list of bundles
        