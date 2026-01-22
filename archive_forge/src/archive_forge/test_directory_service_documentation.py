from .. import transport, urlutils
from ..directory_service import (AliasDirectory, DirectoryServiceRegistry,
from . import TestCase, TestCaseWithTransport
Test compatibility with older implementations of Directory
    that don't support the purpose argument.