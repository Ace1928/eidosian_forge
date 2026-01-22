from paste.util import ip4
def make_grantip(app, global_conf, clobber_username=False, **kw):
    """
    Grant roles or usernames based on IP addresses.

    Config looks like this::

      [filter:grant]
      use = egg:Paste#grantip
      clobber_username = true
      # Give localhost system role (no username):
      127.0.0.1 = -:system
      # Give everyone in 192.168.0.* editor role:
      192.168.0.0/24 = -:editor
      # Give one IP the username joe:
      192.168.0.7 = joe
      # And one IP is should not be logged in:
      192.168.0.10 = __remove__:-editor

    """
    from paste.deploy.converters import asbool
    clobber_username = asbool(clobber_username)
    ip_map = {}
    for key, value in kw.items():
        if ':' in value:
            username, role = value.split(':', 1)
        else:
            username = value
            role = ''
        if username == '-':
            username = ''
        if role == '-':
            role = ''
        ip_map[key] = value
    return GrantIPMiddleware(app, ip_map, clobber_username)