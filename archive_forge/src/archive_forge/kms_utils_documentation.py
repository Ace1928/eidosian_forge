from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
Gets the Cloud KMS CryptoKey reference for a boot disk's initialize params.

  Args:
    args: A dictionary of a boot disk's initialize params.
    messages: Compute API messages module.
    current_value: Current CustomerEncryptionKey value.
    conflicting_arg: name of conflicting argument

  Returns:
    CustomerEncryptionKey message with the KMS key populated if args has a key.
  Raises:
    ConflictingArgumentsException if an encryption key is already populated.
  