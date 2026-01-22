import keystone.conf
def symptom_unreasonable_max_token_size():
    """`keystone.conf [DEFAULT] max_token_size` should be adjusted.

    This option is intended to protect keystone from unreasonably sized tokens,
    where "reasonable" is mostly dependent on the `keystone.conf [token]
    provider` that you're using. If you're using one of the following token
    providers, then you should set `keystone.conf [DEFAULT] max_token_size`
    accordingly:

    - For Fernet, set `keystone.conf [DEFAULT] max_token_size = 255`, because
      Fernet tokens should never exceed this length in most deployments.
      However, if you are also using `keystone.conf [identity] driver = ldap`,
      Fernet tokens may not be built using an efficient packing method,
      depending on the IDs returned from LDAP, resulting in longer Fernet
      tokens (adjust your `max_token_size` accordingly).
    """
    return 'fernet' in CONF.token.provider and CONF.max_token_size > 255