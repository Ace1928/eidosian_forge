import base64
import contextlib
import gzip
import json
import os
import shutil
import subprocess
import tempfile
import typing as ty
@contextlib.contextmanager
def populate_directory(metadata, user_data=None, versions=None, network_data=None, vendor_data=None):
    """Populate a directory with configdrive files.

    :param dict metadata: Metadata.
    :param bytes user_data: Vendor-specific user data.
    :param versions: List of metadata versions to support.
    :param dict network_data: Networking configuration.
    :param dict vendor_data: Extra supplied vendor data.
    :return: a context manager yielding a directory with files
    """
    d = tempfile.mkdtemp()
    versions = versions or ('2012-08-10', 'latest')
    try:
        for version in versions:
            subdir = os.path.join(d, 'openstack', version)
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            with open(os.path.join(subdir, 'meta_data.json'), 'w') as fp:
                json.dump(metadata, fp)
            if network_data:
                with open(os.path.join(subdir, 'network_data.json'), 'w') as fp:
                    json.dump(network_data, fp)
            if vendor_data:
                with open(os.path.join(subdir, 'vendor_data2.json'), 'w') as fp:
                    json.dump(vendor_data, fp)
            if user_data:
                flag = 't' if isinstance(user_data, str) else 'b'
                with open(os.path.join(subdir, 'user_data'), 'w%s' % flag) as fp:
                    fp.write(user_data)
        yield d
    finally:
        shutil.rmtree(d)