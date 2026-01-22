from __future__ import absolute_import, division, print_function
def identify_private_key_format(content, encoding='utf-8'):
    """Given the contents of a private key file, identifies its format."""
    try:
        first_pem = extract_first_pem(content.decode(encoding))
        if first_pem is None:
            return 'raw'
        lines = first_pem.splitlines(False)
        if lines[0].startswith(PEM_START) and lines[0].endswith(PEM_END) and (len(lines[0]) > len(PEM_START) + len(PEM_END)):
            name = lines[0][len(PEM_START):-len(PEM_END)]
            if name in PKCS8_PRIVATEKEY_NAMES:
                return 'pkcs8'
            if len(name) > len(PKCS1_PRIVATEKEY_SUFFIX) and name.endswith(PKCS1_PRIVATEKEY_SUFFIX):
                return 'pkcs1'
            return 'unknown-pem'
    except UnicodeDecodeError:
        pass
    return 'raw'