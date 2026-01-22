from troveclient.compat import exceptions
def get_authenticator_cls(cls_or_name):
    """Factory method to retrieve Authenticator class."""
    if isinstance(cls_or_name, type):
        return cls_or_name
    elif isinstance(cls_or_name, str):
        if cls_or_name == 'keystone':
            return KeyStoneV3Authenticator
        elif cls_or_name == 'auth1.1':
            return Auth1_1
        elif cls_or_name == 'fake':
            return FakeAuth
    raise ValueError('Could not determine authenticator class from the given value %r.' % cls_or_name)