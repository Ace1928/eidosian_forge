import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.kms import exceptions
from boto.compat import six
import base64
def generate_data_key(self, key_id, encryption_context=None, number_of_bytes=None, key_spec=None, grant_tokens=None):
    """
        Generates a secure data key. Data keys are used to encrypt and
        decrypt data. They are wrapped by customer master keys.

        :type key_id: string
        :param key_id: Unique identifier of the key. This can be an ARN, an
            alias, or a globally unique identifier.

        :type encryption_context: map
        :param encryption_context: Name/value pair that contains additional
            data to be authenticated during the encryption and decryption
            processes that use the key. This value is logged by AWS CloudTrail
            to provide context around the data encrypted by the key.

        :type number_of_bytes: integer
        :param number_of_bytes: Integer that contains the number of bytes to
            generate. Common values are 128, 256, 512, 1024 and so on. 1024 is
            the current limit.

        :type key_spec: string
        :param key_spec: Value that identifies the encryption algorithm and key
            size to generate a data key for. Currently this can be AES_128 or
            AES_256.

        :type grant_tokens: list
        :param grant_tokens: A list of grant tokens that represent grants which
            can be used to provide long term permissions to generate a key.

        """
    params = {'KeyId': key_id}
    if encryption_context is not None:
        params['EncryptionContext'] = encryption_context
    if number_of_bytes is not None:
        params['NumberOfBytes'] = number_of_bytes
    if key_spec is not None:
        params['KeySpec'] = key_spec
    if grant_tokens is not None:
        params['GrantTokens'] = grant_tokens
    response = self.make_request(action='GenerateDataKey', body=json.dumps(params))
    if response.get('CiphertextBlob') is not None:
        response['CiphertextBlob'] = base64.b64decode(response['CiphertextBlob'].encode('utf-8'))
    if response.get('Plaintext') is not None:
        response['Plaintext'] = base64.b64decode(response['Plaintext'].encode('utf-8'))
    return response