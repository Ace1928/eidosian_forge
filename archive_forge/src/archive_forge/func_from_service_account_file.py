import abc
import io
import json
import six
from google.auth import exceptions
@classmethod
def from_service_account_file(cls, filename):
    """Creates a Signer instance from a service account .json file
        in Google format.

        Args:
            filename (str): The path to the service account .json file.

        Returns:
            google.auth.crypt.Signer: The constructed signer.
        """
    with io.open(filename, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    return cls.from_service_account_info(data)