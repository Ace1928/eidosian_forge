import sys
from jsonschema.tests._suite import Suite
import jsonschema
def complex_email_validation(test):
    if test.subject != 'email':
        return
    message = 'Complex email validation is (intentionally) unsupported.'
    return skip(message=message, description='an invalid domain')(test) or skip(message=message, description='an invalid IPv4-address-literal')(test) or skip(message=message, description='dot after local part is not valid')(test) or skip(message=message, description='dot before local part is not valid')(test) or skip(message=message, description='two subsequent dots inside local part are not valid')(test)