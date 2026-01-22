import re
import keystone.conf
def symptom_invalid_password_regular_expression():
    """Invalid password regular expression.

    The password regular expression is invalid and users will not be able to
    make password changes until this has been corrected.

    Ensure `[security_compliance] password_regex` is a valid regular
    expression.
    """
    try:
        if CONF.security_compliance.password_regex:
            re.match(CONF.security_compliance.password_regex, 'password')
        return False
    except re.error:
        return True