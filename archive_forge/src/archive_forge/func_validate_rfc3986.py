import re
def validate_rfc3986(url, rule='URI'):
    """
    Validates strings according to RFC3986

    :param url: String cointaining URI to validate
    :param rule: It could be 'URI' (default) or 'URI_reference'.
    :return: True or False
    """
    if rule == 'URI':
        return URI_RE_COMP.match(url)
    elif rule == 'URI_reference':
        return URI_REF_RE_COMP.match(url)
    else:
        raise ValueError('Invalid rule')