import base64
import datetime
from os import remove
from os.path import join
from OpenSSL import crypto
import dateutil.parser
import pytz
import saml2.cryptography.pki
def read_str_from_file(self, file, type='pem'):
    with open(file, 'rb') as f:
        str_data = f.read()
    if type == 'pem':
        return str_data
    if type in ['der', 'cer', 'crt']:
        return base64.b64encode(str(str_data))