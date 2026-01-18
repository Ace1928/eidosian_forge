import re
from typing import Type, get_args, get_origin

# ──────────────────────────────────────────────────────────────
# String Formatting Functions
# ──────────────────────────────────────────────────────────────


def format_message(subject: str, value: int) -> str:
    """
    Format a message by combining a string subject with an integer value.

    Creates a precisely formatted string combining the subject and value
    in a grammatically coherent structure.

    Args:
        subject (str): The subject of the message. Must be a valid string.
        value (int): The numerical value associated with the subject. Must be an integer.

    Returns:
        str: A formatted message combining the subject and value.

    Examples:
        >>> format_message("Temperature", 72)
        'Temperature has a value of 72.'

    Note:
        This function performs no validation on the inputs beyond typing.
    """
    return f"{subject} has a value of {value}."


def format_type_name(typ: Type[object]) -> str:
    """
    Format a type into a clean, readable string representation.

    Converts Python type objects into standardized string representations
    for consistent display and logging. Handles generics, unions, and nested types.

    Args:
        typ (Type[object]): The type to format into a string

    Returns:
        str: A standardized string representation of the type

    Examples:
        >>> format_type_name(int)
        'int'
        >>> format_type_name(List[int])  # doctest: +SKIP
        'List[int]'
        >>> format_type_name(Dict[str, Optional[int]])  # doctest: +SKIP
        'Dict[str, Optional[int]]'

    Note:
        This function handles generic types, union types, and nested types.
    """
    if typ is type(None):
        return "None"

    # Use Python 3.8+ typing utilities
    origin = get_origin(typ)
    args = get_args(typ)

    if origin is not None and args:
        # Handle generic types
        origin_name = getattr(origin, "__name__", str(origin))
        args_str = ", ".join(format_type_name(arg) for arg in args)
        return f"{origin_name}[{args_str}]"

    # Handle regular types
    if hasattr(typ, "__name__"):
        return typ.__name__

    return str(typ)


def camel_to_snake(name: str) -> str:
    """
    Convert a camelCase or PascalCase string to snake_case.

    Transforms identifiers from camelCase or PascalCase convention to
    snake_case for consistent naming across the system.

    Args:
        name (str): The camelCase or PascalCase string to convert

    Returns:
        str: The equivalent snake_case string

    Examples:
        >>> camel_to_snake("camelCase")
        'camel_case'
        >>> camel_to_snake("PascalCase")
        'pascal_case'
        >>> camel_to_snake("HTTPResponse")
        'http_response'
        >>> camel_to_snake("")
        ''

    Note:
        Handles special cases with consecutive uppercase letters correctly.
    """
    if not name:
        return name

    # Use regex to properly handle acronyms and consecutive uppercase letters
    # Convert patterns like 'HTTPResponse' to 'http_response' correctly
    pattern = re.compile(r"([A-Z]?[a-z]+)|([A-Z]+(?=[A-Z][a-z]|\b))")
    result = "_".join([m.group(0).lower() for m in pattern.finditer(name)])
    return result


def snake_to_camel(name: str) -> str:
    """
    Convert a snake_case string to camelCase.

    Transforms identifiers from snake_case convention to camelCase
    for consistent naming across the system.

    Args:
        name (str): The snake_case string to convert

    Returns:
        str: The equivalent camelCase string

    Examples:
        >>> snake_to_camel("snake_case")
        'snakeCase'
        >>> snake_to_camel("http_response")
        'httpResponse'
        >>> snake_to_camel("__private_var")
        '__privateVar'
        >>> snake_to_camel("")
        ''

    Note:
        Preserves leading underscores for private/protected identifiers.
    """
    # Extract leading underscores to preserve them
    leading_underscores = ""
    while name and name[0] == "_":
        leading_underscores += "_"
        name = name[1:]

    # Handle empty string or string with only underscores
    if not name:
        return leading_underscores

    # Split by underscore and capitalize each word except the first
    components = name.split("_")
    camel = components[0].lower() + "".join(
        word.title() for word in components[1:] if word
    )

    return leading_underscores + camel


def pluralize(word: str) -> str:
    """
    Convert a singular word to its plural form using simple English rules.

    Applies common English pluralization rules to transform singular nouns
    to their plural forms.

    Args:
        word (str): The singular word to pluralize

    Returns:
        str: The pluralized form of the word

    Examples:
        >>> pluralize("cat")
        'cats'
        >>> pluralize("class")
        'classes'
        >>> pluralize("study")
        'studies'
        >>> pluralize("box")
        'boxes'
        >>> pluralize("child")  # Special cases are not handled
        'childs'

    Note:
        Handles common English pluralization rules but not irregular nouns.
    """
    if not word:
        return word

    # Common English pluralization rules
    if word.endswith(("s", "sh", "ch", "x", "z")):
        return word + "es"
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


def standardize_typename(name: str) -> str:
    """
    Standardize a type name for consistent representation.

    Converts various representations of type names to a standardized format,
    handling common variations and aliases.

    Args:
        name (str): The type name to standardize

    Returns:
        str: The standardized type name

    Examples:
        >>> standardize_typename("int")
        'int'
        >>> standardize_typename("integer")
        'int'
        >>> standardize_typename("str")
        'str'
        >>> standardize_typename("string")
        'str'
        >>> standardize_typename("bool")
        'bool'
        >>> standardize_typename("boolean")
        'bool'

    Note:
        Normalizes common type name variations to their Python equivalents.
    """
    # Dictionary of common type name variations mapped to standardized names
    type_aliases = {
        "integer": "int",
        "string": "str",
        "boolean": "bool",
        "floating point": "float",
        "floating-point": "float",
        "double": "float",
        "dictionary": "dict",
        "mapping": "dict",
        "sequence": "list",
        "array": "list",
        "none": "NoneType",
        "null": "NoneType",
        "nothing": "NoneType",
        "undefined": "NoneType",
    }

    # Normalize input (lowercase and strip whitespace)
    normalized = name.lower().strip()

    # Return the standardized name if found, otherwise return the original
    return type_aliases.get(normalized, name)
