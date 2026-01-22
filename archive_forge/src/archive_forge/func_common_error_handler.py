import functools
from .exceptions import AnsibleAWSError
@classmethod
def common_error_handler(cls, description):
    """A simple error handler that catches the standard Boto3 exceptions and raises
        an AnsibleAWSError exception.

        param: description: a description of the action being taken.
                            Exception raised will include a message of
                            f"Timeout trying to {description}" or
                            f"Failed to {description}"
        """

    def wrapper(func):

        @functools.wraps(func)
        def handler(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except botocore.exceptions.WaiterError as e:
                raise cls._CUSTOM_EXCEPTION(message=f'Timeout trying to {description}', exception=e) from e
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                raise cls._CUSTOM_EXCEPTION(message=f'Failed to {description}', exception=e) from e
        return handler
    return wrapper