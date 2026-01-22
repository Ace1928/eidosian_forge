import base64
import errno
import hashlib
import logging
import zlib
from debtcollector import removals
from keystoneclient import exceptions
from keystoneclient.i18n import _
def cms_sign_data(data_to_sign, signing_cert_file_name, signing_key_file_name, outform=PKI_ASN1_FORM, message_digest=DEFAULT_TOKEN_DIGEST_ALGORITHM):
    """Use OpenSSL to sign a document.

    Produces a Base64 encoding of a DER formatted CMS Document
    http://en.wikipedia.org/wiki/Cryptographic_Message_Syntax

    :param data_to_sign: data to sign
    :param signing_cert_file_name:  path to the X509 certificate containing
        the public key associated with the private key used to sign the data
    :param signing_key_file_name: path to the private key used to sign
        the data
    :param outform: Format for the signed document PKIZ_CMS_FORM or
        PKI_ASN1_FORM
    :param message_digest: Digest algorithm to use when signing or resigning

    """
    _ensure_subprocess()
    if isinstance(data_to_sign, str):
        data = bytes(data_to_sign, encoding='utf-8')
    else:
        data = data_to_sign
    process = subprocess.Popen(['openssl', 'cms', '-sign', '-signer', signing_cert_file_name, '-inkey', signing_key_file_name, '-outform', 'PEM', '-nosmimecap', '-nodetach', '-nocerts', '-noattr', '-md', message_digest], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
    output, err, retcode = _process_communicate_handle_oserror(process, data, (signing_cert_file_name, signing_key_file_name))
    if retcode != OpensslCmsExitStatus.SUCCESS or 'Error' in err:
        if retcode == OpensslCmsExitStatus.CREATE_CMS_READ_MIME_ERROR:
            LOG.error('Signing error: Unable to load certificate - ensure you have configured PKI with "keystone-manage pki_setup"')
        else:
            LOG.error('Signing error: %s', err)
        raise subprocess.CalledProcessError(retcode, 'openssl')
    if outform == PKI_ASN1_FORM:
        return output.decode('utf-8')
    else:
        return output