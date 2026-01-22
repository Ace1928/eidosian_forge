import nacl.public
def generate_key():
    """GenerateKey generates a new PrivateKey.
    :return: a PrivateKey
    """
    return PrivateKey(nacl.public.PrivateKey.generate())