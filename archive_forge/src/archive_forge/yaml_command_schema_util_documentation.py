from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import unicode_literals
import abc
from collections.abc import Callable
import dataclasses
from typing import Any
from apitools.base.protorpclite import messages as apitools_messages
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import arg_parsers_usage_text as usage_text
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import module_util
Generates an argparse type function to use to parse the argument.

    The return of the type function will be a list of instances of the given
    message with the fields filled in.

    Args:
      field: apitools field instance we are generating ArgObject for

    Raises:
      InvalidSchemaError: If a type for a field could not be determined.

    Returns:
      _AdditionalPropsType, The type function that parses the ArgDict
        and returns a list of message instances.
    