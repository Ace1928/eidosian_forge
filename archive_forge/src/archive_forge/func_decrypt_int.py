from rsa._compat import is_integer
def decrypt_int(cyphertext, dkey, n):
    """Decrypts a cypher text using the decryption key 'dkey', working modulo n"""
    assert_int(cyphertext, 'cyphertext')
    assert_int(dkey, 'dkey')
    assert_int(n, 'n')
    message = pow(cyphertext, dkey, n)
    return message