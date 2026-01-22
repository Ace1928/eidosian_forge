import random
import string
def random_tsigkey_secret(name='test-secret'):
    return f'{name}-{random_digits(254 - len(name))}'