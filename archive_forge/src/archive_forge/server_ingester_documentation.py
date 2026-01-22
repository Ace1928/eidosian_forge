import errno
import logging
import os
import subprocess
import tempfile
import time
import grpc
import pkg_resources
from tensorboard.data import grpc_provider
from tensorboard.data import ingester
from tensorboard.data.proto import data_provider_pb2
from tensorboard.util import tb_logging
Test whether the binary's version is at least the given one.

        Useful for gating features that are available in the latest data
        server builds from head, but not yet released to PyPI. For
        example, if v0.4.0 is the latest published version, you can
        check `at_least_version("0.5.0a0")` to include both prereleases
        at head and the eventual final release of v0.5.0.

        If this binary's version was set to `None` at construction time,
        this method always returns `True`.

        Args:
          required_version: PEP 396-compliant version string.

        Returns:
          Boolean.
        