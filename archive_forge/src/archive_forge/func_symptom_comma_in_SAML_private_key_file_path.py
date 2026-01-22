import keystone.conf
def symptom_comma_in_SAML_private_key_file_path():
    """`[saml] certfile` should not contain a comma (`,`).

    Because a comma is part of the API between keystone and the external
    xmlsec1 binary which utilizes the key, keystone cannot include a comma in
    the path to the private key file.
    """
    return ',' in CONF.saml.keyfile