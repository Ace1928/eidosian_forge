from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import missing_required_lib  # pylint: disable=unused-import:
from ansible.module_utils.six.moves import configparser
from ansible.module_utils._text import to_native
import traceback
import os
import ssl as ssl_lib
def rename_ssl_option_for_pymongo4(connection_options):
    """
    This function renames the old ssl parameter, and sorts the data out,
    when the driver use is >= PyMongo 4
    """
    if int(PyMongoVersion[0]) >= 4:
        if connection_options.get('ssl_cert_reqs', None) in ('CERT_NONE', ssl_lib.CERT_NONE):
            connection_options['tlsAllowInvalidCertificates'] = True
        elif connection_options.get('ssl_cert_reqs', None) in ('CERT_REQUIRED', ssl_lib.CERT_REQUIRED):
            connection_options['tlsAllowInvalidCertificates'] = False
        connection_options.pop('ssl_cert_reqs', None)
        if connection_options.get('ssl_ca_certs', None) is not None:
            connection_options['tlsCAFile'] = connection_options['ssl_ca_certs']
        connection_options.pop('ssl_ca_certs', None)
        connection_options.pop('ssl_crlfile', None)
        if connection_options.get('ssl_certfile', None) is not None:
            connection_options['tlsCertificateKeyFile'] = connection_options['ssl_certfile']
        elif connection_options.get('ssl_keyfile', None) is not None:
            connection_options['tlsCertificateKeyFile'] = connection_options['ssl_keyfile']
        connection_options.pop('ssl_certfile', None)
        connection_options.pop('ssl_keyfile', None)
        if connection_options.get('ssl_pem_passphrase', None) is not None:
            connection_options['tlsCertificateKeyFilePassword'] = connection_options['ssl_pem_passphrase']
        connection_options.pop('ssl_pem_passphrase', None)
    return connection_options