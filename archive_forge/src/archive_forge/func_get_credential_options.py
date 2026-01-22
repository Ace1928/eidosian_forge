import logging
import os
from types import SimpleNamespace
from rasterio._path import _parse_path, _UnparsedPath
def get_credential_options(self):
    """Get credentials as GDAL configuration options

        Returns
        -------
        dict

        """
    if self.unsigned:
        return {'AZURE_NO_SIGN_REQUEST': 'YES', 'AZURE_STORAGE_ACCOUNT': self.storage_account}
    else:
        return {k.upper(): v for k, v in self.credentials.items()}