import os
import string
def authn_request(**kwargs):
    kwargs.setdefault('issuer', 'https://openstack4.local/Shibboleth.sso/SAML2/ECP')
    return template('authn_request.xml', **kwargs).encode('utf-8')