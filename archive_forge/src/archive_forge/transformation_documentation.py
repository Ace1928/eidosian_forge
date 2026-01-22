from ansible.module_utils.six import integer_types
from ansible.module_utils.six import string_types

    Iterate over a dictionary removing any keys that have a None value

    Reference: https://github.com/ansible-collections/community.aws/issues/251
    Credit: https://medium.com/better-programming/how-to-remove-null-none-values-from-a-dictionary-in-python-1bedf1aab5e4

    :param descend_into_lists: whether or not to descend in to lists to continue to remove None values
    :param parameters: parameter dict
    :return: parameter dict with all keys = None removed
    