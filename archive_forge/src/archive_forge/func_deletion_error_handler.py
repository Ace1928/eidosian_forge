import functools
from .exceptions import AnsibleAWSError
@classmethod
def deletion_error_handler(cls, description):
    """A simple error handler that catches the standard Boto3 exceptions and raises
        an AnsibleAWSError exception.
        Error codes representing a non-existent entity will result in None being returned
        Generally used in deletion calls where NoSuchEntity means it's already gone

        param: description: a description of the action being taken.
                            Exception raised will include a message of
                            f"Timeout trying to {description}" or
                            f"Failed to {description}"
        """

    def wrapper(func):

        @functools.wraps(func)
        @cls.common_error_handler(description)
        def handler(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except cls._is_missing():
                return False
        return handler
    return wrapper