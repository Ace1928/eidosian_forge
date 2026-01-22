import base64
import errno
import hashlib
import logging
import zlib
from debtcollector import removals
from keystoneclient import exceptions
from keystoneclient.i18n import _
def cms_verify(formatted, signing_cert_file_name, ca_file_name, inform=PKI_ASN1_FORM):
    """Verify the signature of the contents IAW CMS syntax.

    :raises subprocess.CalledProcessError:
    :raises keystoneclient.exceptions.CertificateConfigError: if certificate
                                                              is not configured
                                                              properly.
    """
    _ensure_subprocess()
    if isinstance(formatted, str):
        data = bytes(formatted, _encoding_for_form(inform))
    else:
        data = formatted
    process = subprocess.Popen(['openssl', 'cms', '-verify', '-certfile', signing_cert_file_name, '-CAfile', ca_file_name, '-inform', 'PEM', '-nosmimecap', '-nodetach', '-nocerts', '-noattr'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
    output, err, retcode = _process_communicate_handle_oserror(process, data, (signing_cert_file_name, ca_file_name))
    if retcode == OpensslCmsExitStatus.INPUT_FILE_READ_ERROR:
        if err.startswith('Error reading S/MIME message'):
            raise exceptions.CMSError(err)
        else:
            raise exceptions.CertificateConfigError(err)
    elif retcode == OpensslCmsExitStatus.COMMAND_OPTIONS_PARSING_ERROR:
        if err.startswith('cms: Cannot open input file'):
            raise exceptions.CertificateConfigError(err)
        else:
            raise subprocess.CalledProcessError(retcode, 'openssl', output=err)
    elif retcode != OpensslCmsExitStatus.SUCCESS:
        raise subprocess.CalledProcessError(retcode, 'openssl', output=err)
    return output