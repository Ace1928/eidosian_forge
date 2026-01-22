from rsa._compat import range
import rsa.common
import rsa.randnum
def is_prime(number):
    """Returns True if the number is prime, and False otherwise.

    >>> is_prime(2)
    True
    >>> is_prime(42)
    False
    >>> is_prime(41)
    True
    """
    if number < 10:
        return number in {2, 3, 5, 7}
    if not number & 1:
        return False
    k = get_primality_testing_rounds(number)
    return miller_rabin_primality_testing(number, k + 1)