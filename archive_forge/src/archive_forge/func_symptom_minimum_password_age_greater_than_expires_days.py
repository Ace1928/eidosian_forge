import re
import keystone.conf
def symptom_minimum_password_age_greater_than_expires_days():
    """Minimum password age should be less than the password expires days.

    If the minimum password age is greater than or equal to the password
    expires days, then users would not be able to change their passwords before
    they expire.

    Ensure `[security_compliance] minimum_password_age` is less than the
    `[security_compliance] password_expires_days`.
    """
    min_age = CONF.security_compliance.minimum_password_age
    expires = CONF.security_compliance.password_expires_days
    return min_age >= expires if min_age > 0 and expires > 0 else False