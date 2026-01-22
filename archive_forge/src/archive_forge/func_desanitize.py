from typing import Dict, Union
def desanitize(sanitized_text: str, secure_context: bytes) -> str:
    """
    Restore the original sensitive data from the sanitized text.

    Args:
        sanitized_text: Sanitized text.
        secure_context: Secure context returned by the `sanitize` function.

    Returns:
        De-sanitized text.
    """
    try:
        import opaqueprompts as op
    except ImportError:
        raise ImportError('Could not import the `opaqueprompts` Python package, please install it with `pip install opaqueprompts`.')
    desanitize_response: op.DesanitizeResponse = op.desanitize(sanitized_text, secure_context)
    return desanitize_response.desanitized_text