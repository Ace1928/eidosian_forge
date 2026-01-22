import functools
from .exceptions import AnsibleAWSError
@classmethod
def list_error_handler(cls, description, default_value=None):
    """A simple error handler that catches the standard Boto3 exceptions and raises
        an AnsibleAWSError exception.
        Error codes representing a non-existent entity will result in None being returned
        Generally used for Get/List calls where the exception just means the resource isn't there

        param: description: a description of the action being taken.
                            Exception raised will include a message of
                            f"Timeout trying to {description}" or
                            f"Failed to {description}"
        param: default_value: the value to return if no matching
                            resources are returned.  Defaults to None
        """

    def wrapper(func):

        @functools.wraps(func)
        @cls.common_error_handler(description)
        def handler(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except cls._is_missing():
                return default_value
        return handler
    return wrapper