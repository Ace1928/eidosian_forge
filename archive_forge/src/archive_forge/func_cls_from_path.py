import logging
import os
from types import SimpleNamespace
from rasterio._path import _parse_path, _UnparsedPath
@staticmethod
def cls_from_path(path):
    """Find the session class suited to the data at `path`.

        Parameters
        ----------
        path : str
            A dataset path or identifier.

        Returns
        -------
        class

        """
    if not path:
        return DummySession
    path = _parse_path(path)
    if isinstance(path, _UnparsedPath) or path.is_local:
        return DummySession
    elif (path.scheme == 's3' or 'amazonaws.com' in path.path) and (not 'X-Amz-Signature' in path.path):
        if boto3 is not None:
            return AWSSession
        else:
            log.info('boto3 not available, falling back to a DummySession.')
            return DummySession
    elif path.scheme == 'oss' or 'aliyuncs.com' in path.path:
        return OSSSession
    elif path.path.startswith('/vsiswift/'):
        return SwiftSession
    elif path.scheme == 'az':
        return AzureSession
    else:
        return DummySession