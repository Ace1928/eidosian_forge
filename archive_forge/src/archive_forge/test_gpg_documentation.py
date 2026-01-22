import sys
from io import BytesIO
from .. import config, errors, gpg, tests, trace, ui
from . import TestCase, features
Tests for signing and verifying blobs of data via gpg.