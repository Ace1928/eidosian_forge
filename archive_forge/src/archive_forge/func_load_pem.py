import base64
from rsa._compat import is_bytes, range
def load_pem(contents, pem_marker):
    """Loads a PEM file.

    :param contents: the contents of the file to interpret
    :param pem_marker: the marker of the PEM content, such as 'RSA PRIVATE KEY'
        when your file has '-----BEGIN RSA PRIVATE KEY-----' and
        '-----END RSA PRIVATE KEY-----' markers.

    :return: the base64-decoded content between the start and end markers.

    @raise ValueError: when the content is invalid, for example when the start
        marker cannot be found.

    """
    if not is_bytes(contents):
        contents = contents.encode('ascii')
    pem_start, pem_end = _markers(pem_marker)
    pem_lines = []
    in_pem_part = False
    for line in contents.splitlines():
        line = line.strip()
        if not line:
            continue
        if line == pem_start:
            if in_pem_part:
                raise ValueError('Seen start marker "%s" twice' % pem_start)
            in_pem_part = True
            continue
        if not in_pem_part:
            continue
        if in_pem_part and line == pem_end:
            in_pem_part = False
            break
        if b':' in line:
            continue
        pem_lines.append(line)
    if not pem_lines:
        raise ValueError('No PEM start marker "%s" found' % pem_start)
    if in_pem_part:
        raise ValueError('No PEM end marker "%s" found' % pem_end)
    pem = b''.join(pem_lines)
    return base64.standard_b64decode(pem)