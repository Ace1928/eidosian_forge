# Enhanced Documentation and Commenting for log_function_call decorator


def log_function_call(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to log function calls, including entry, exit, arguments, and exceptions.

    This decorator enhances debugging and monitoring by logging detailed insights into
    the function's execution flow, including the function name, arguments passed, return
    value, and any exceptions raised.

    Args:
        func (Callable[..., Any]): The function to be decorated.

    Returns:
        Callable[..., Any]: The decorated function with added logging functionality.

    Example Usage:
        @log_function_call
        def example_function(param1, param2):
            return param1 + param2
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Wraps the original function, adding logging for entry, exit, arguments, and exceptions.
        args_repr = [
            repr(a) for a in args
        ]  # List comprehension to create string representations of positional arguments.
        kwargs_repr = [
            f"{k}={v!r}" for k, v in kwargs.items()
        ]  # List comprehension for keyword arguments.
        signature = ", ".join(
            args_repr + kwargs_repr
        )  # Combine all argument representations into a single string.

        # Log the function call with its signature.
        logging.debug(f"Calling {func.__name__}({signature})")

        try:
            # Execute the original function with provided arguments and capture the result.
            result = func(*args, **kwargs)
            # Log the function's return value.
            logging.debug(f"{func.__name__} returned {result!r}")
            return result
        except Exception as e:
            # Log any exception raised by the function, including the exception type and message.
            logging.error(f"{func.__name__} raised {e.__class__.__name__}: {e}")
            # Log the exception for debugging purposes. Reraising allows external error handling to address the exception.
            raise  # Reraise the exception for further handling.

    return wrapper
