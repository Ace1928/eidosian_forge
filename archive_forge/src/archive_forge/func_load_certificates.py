import datetime
import glob
import os
from typing import Dict, Optional, Tuple, Union
import zmq
def load_certificates(directory: Union[str, os.PathLike]='.') -> Dict[bytes, bool]:
    """Load public keys from all certificates in a directory"""
    certs = {}
    if not os.path.isdir(directory):
        raise OSError(f'Invalid certificate directory: {directory}')
    glob_string = os.path.join(directory, '*.key')
    cert_files = glob.glob(glob_string)
    for cert_file in cert_files:
        public_key, _ = load_certificate(cert_file)
        if public_key:
            certs[public_key] = True
    return certs