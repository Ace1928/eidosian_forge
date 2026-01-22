import datetime
import os
import socket
def load_certs_from_env():
    tls_env_vars = ['RAY_TLS_SERVER_CERT', 'RAY_TLS_SERVER_KEY', 'RAY_TLS_CA_CERT']
    if any((v not in os.environ for v in tls_env_vars)):
        raise RuntimeError('If the environment variable RAY_USE_TLS is set to true then RAY_TLS_SERVER_CERT, RAY_TLS_SERVER_KEY and RAY_TLS_CA_CERT must also be set.')
    with open(os.environ['RAY_TLS_SERVER_CERT'], 'rb') as f:
        server_cert_chain = f.read()
    with open(os.environ['RAY_TLS_SERVER_KEY'], 'rb') as f:
        private_key = f.read()
    with open(os.environ['RAY_TLS_CA_CERT'], 'rb') as f:
        ca_cert = f.read()
    return (server_cert_chain, private_key, ca_cert)