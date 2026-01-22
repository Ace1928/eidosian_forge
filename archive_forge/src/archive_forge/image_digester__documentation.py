from __future__ import absolute_import
from __future__ import print_function
import argparse
import logging
import sys
from containerregistry.client.v2_2 import docker_image as v2_2_image
from containerregistry.client.v2_2 import oci_compat
from containerregistry.tools import logging_setup
from six.moves import zip  # pylint: disable=redefined-builtin
This package calculates the digest of an image.

The format this tool *expects* to deal with is proprietary.
Image digests aren't stable upon gzip implementation/configuration.
This tool is expected to be only self-consistent.
