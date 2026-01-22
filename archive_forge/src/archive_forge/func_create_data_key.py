from __future__ import annotations
import contextlib
import enum
import socket
import weakref
from copy import deepcopy
from typing import (
from bson import _dict_to_bson, decode, encode
from bson.binary import STANDARD, UUID_SUBTYPE, Binary
from bson.codec_options import CodecOptions
from bson.errors import BSONError
from bson.raw_bson import DEFAULT_RAW_BSON_OPTIONS, RawBSONDocument, _inflate_bson
from bson.son import SON
from pymongo import _csot
from pymongo.collection import Collection
from pymongo.common import CONNECT_TIMEOUT
from pymongo.cursor import Cursor
from pymongo.daemon import _spawn_daemon
from pymongo.database import Database
from pymongo.encryption_options import AutoEncryptionOpts, RangeOpts
from pymongo.errors import (
from pymongo.mongo_client import MongoClient
from pymongo.network import BLOCKING_IO_ERRORS
from pymongo.operations import UpdateOne
from pymongo.pool import PoolOptions, _configured_socket, _raise_connection_failure
from pymongo.read_concern import ReadConcern
from pymongo.results import BulkWriteResult, DeleteResult
from pymongo.ssl_support import get_ssl_context
from pymongo.typings import _DocumentType, _DocumentTypeArg
from pymongo.uri_parser import parse_host
from pymongo.write_concern import WriteConcern
def create_data_key(self, kms_provider: str, master_key: Optional[Mapping[str, Any]]=None, key_alt_names: Optional[Sequence[str]]=None, key_material: Optional[bytes]=None) -> Binary:
    """Create and insert a new data key into the key vault collection.

        :Parameters:
          - `kms_provider`: The KMS provider to use. Supported values are
            "aws", "azure", "gcp", "kmip", and "local".
          - `master_key`: Identifies a KMS-specific key used to encrypt the
            new data key. If the kmsProvider is "local" the `master_key` is
            not applicable and may be omitted.

            If the `kms_provider` is "aws" it is required and has the
            following fields::

              - `region` (string): Required. The AWS region, e.g. "us-east-1".
              - `key` (string): Required. The Amazon Resource Name (ARN) to
                 the AWS customer.
              - `endpoint` (string): Optional. An alternate host to send KMS
                requests to. May include port number, e.g.
                "kms.us-east-1.amazonaws.com:443".

            If the `kms_provider` is "azure" it is required and has the
            following fields::

              - `keyVaultEndpoint` (string): Required. Host with optional
                 port, e.g. "example.vault.azure.net".
              - `keyName` (string): Required. Key name in the key vault.
              - `keyVersion` (string): Optional. Version of the key to use.

            If the `kms_provider` is "gcp" it is required and has the
            following fields::

              - `projectId` (string): Required. The Google cloud project ID.
              - `location` (string): Required. The GCP location, e.g. "us-east1".
              - `keyRing` (string): Required. Name of the key ring that contains
                the key to use.
              - `keyName` (string): Required. Name of the key to use.
              - `keyVersion` (string): Optional. Version of the key to use.
              - `endpoint` (string): Optional. Host with optional port.
                Defaults to "cloudkms.googleapis.com".

            If the `kms_provider` is "kmip" it is optional and has the
            following fields::

              - `keyId` (string): Optional. `keyId` is the KMIP Unique
                Identifier to a 96 byte KMIP Secret Data managed object. If
                keyId is omitted, the driver creates a random 96 byte KMIP
                Secret Data managed object.
              - `endpoint` (string): Optional. Host with optional
                 port, e.g. "example.vault.azure.net:".

          - `key_alt_names` (optional): An optional list of string alternate
            names used to reference a key. If a key is created with alternate
            names, then encryption may refer to the key by the unique alternate
            name instead of by ``key_id``. The following example shows creating
            and referring to a data key by alternate name::

              client_encryption.create_data_key("local", key_alt_names=["name1"])
              # reference the key with the alternate name
              client_encryption.encrypt("457-55-5462", key_alt_name="name1",
                                        algorithm=Algorithm.AEAD_AES_256_CBC_HMAC_SHA_512_Random)
          - `key_material` (optional): Sets the custom key material to be used
            by the data key for encryption and decryption.

        :Returns:
          The ``_id`` of the created data key document as a
          :class:`~bson.binary.Binary` with subtype
          :data:`~bson.binary.UUID_SUBTYPE`.

        .. versionchanged:: 4.2
           Added the `key_material` parameter.
        """
    self._check_closed()
    with _wrap_encryption_errors():
        return cast(Binary, self._encryption.create_data_key(kms_provider, master_key=master_key, key_alt_names=key_alt_names, key_material=key_material))