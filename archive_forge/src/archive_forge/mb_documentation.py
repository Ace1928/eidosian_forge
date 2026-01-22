from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import re
import textwrap
from gslib.cloud_api import AccessDeniedException, BadRequestException
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.commands.rpo import VALID_RPO_VALUES
from gslib.commands.rpo import VALID_RPO_VALUES_STRING
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.exception import InvalidUrlError
from gslib.storage_url import StorageUrlFromString
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils.constants import NO_MAX
from gslib.utils.retention_util import RetentionInSeconds
from gslib.utils.shim_util import GcloudStorageFlag
from gslib.utils.shim_util import GcloudStorageMap
from gslib.utils.text_util import InsistAscii
from gslib.utils.text_util import InsistOnOrOff
from gslib.utils.text_util import NormalizeStorageClass
from gslib.utils.encryption_helper import ValidateCMEK
Command entry point for the mb command.