import collections
import io
import random as random_module
import string
def generate_password(length, char_classes):
    """Generate a random password.

    The password will be of the specified length, and comprised of characters
    from the specified character classes, which can be generated using the
    :func:`named_char_class` and :func:`special_char_class` functions. Where
    a minimum count is specified in the character class, at least that number
    of characters in the resulting password are guaranteed to be from that
    character class.

    :param length: The length of the password to generate, in characters
    :param char_classes: Iterable over classes of characters from which to
                         generate a password
    """
    char_buffer = io.StringIO()
    all_allowed_chars = set()
    for char_class in char_classes:
        all_allowed_chars |= char_class.allowed_chars
        allowed_chars = tuple(char_class.allowed_chars)
        for i in range(char_class.min_count):
            char_buffer.write(random.choice(allowed_chars))
    combined_chars = tuple(all_allowed_chars)
    for i in range(max(0, length - char_buffer.tell())):
        char_buffer.write(random.choice(combined_chars))
    selected_chars = char_buffer.getvalue()
    char_buffer.close()
    return ''.join(random.sample(selected_chars, length))