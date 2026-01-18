"""
Type Validation Utilities for the Type Forge System
==================================================

This module provides robust type validation utilities that complement Python's
built-in type checking mechanisms with enhanced safety, precision, and elegance.
These functions maintain strict type boundaries while handling edge cases that
might otherwise cause errors in typical validation scenarios.

Core functionalities include:
- Safe class hierarchy checking that won't raise exceptions
- String validation for identifiers and non-empty values
- Collection and numeric type detection
- Protocol instance verification
- Multi-type validation
- Type conversion safety checking
- Type compatibility verification

Each function follows the principle of returning boolean values for validation
rather than raising exceptions, making them suitable for conditional logic
while maintaining optimal performance characteristics.
"""

import inspect
import numbers
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
    get_origin,
)

from type_forge.typing.aliases import ParentSpecType
from type_forge.typing.definitions import ValidationSeverity
from type_forge.typing.protocols import SupportsFloat, SupportsInt
from type_forge.typing.variables import T

from . import __version__  # noqa: F401

# Import the moved functions


version = __version__


class ValidationIssue:
    """
    Detailed representation of a validation issue with context and severity.

    This class encapsulates information about a validation issue, including
    its severity, location, message, and contextual information for debugging
    and correction.

    Attributes:
        severity (ValidationSeverity): The severity level of the issue
        message (str): Human-readable description of the issue
        path (Optional[str]): Path to the location of the issue (e.g., "user.address.city")
        context (Dict[str, object]): Additional contextual information about the issue

    Examples:
        >>> issue = ValidationIssue(ValidationSeverity.ERROR, "Invalid email format")
        >>> issue.severity.name
        'ERROR'
        >>> issue.message
        'Invalid email format'
        >>> detailed = ValidationIssue(
        ...     ValidationSeverity.WARNING,
        ...     "Value outside recommended range",
        ...     path="settings.timeout",
        ...     context={"value": 120, "recommended_max": 60}
        ... )
        >>> detailed.path
        'settings.timeout'
    """

    def __init__(
        self,
        severity: ValidationSeverity,
        message: str,
        path: Optional[str] = None,
        context: Optional[Dict[str, object]] = None,
    ) -> None:
        """
        Initialize a ValidationIssue.

        Args:
            severity: The severity level of the issue
            message: Human-readable description of the issue
            path: Path to the location of the issue (e.g., "user.address.city")
            context: Additional contextual information about the issue
        """
        self.severity: ValidationSeverity = severity
        self.message: str = message
        self.path: Optional[str] = path
        self.context: Dict[str, object] = context or {}

    def is_error(self) -> bool:
        """
        Check if this issue is an error.

        Returns:
            bool: True if this is an error or fatal issue

        Examples:
            >>> issue = ValidationIssue(ValidationSeverity.ERROR, "Invalid input")
            >>> issue.is_error()
            True
            >>> warning = ValidationIssue(ValidationSeverity.WARNING, "Unusual value")
            >>> warning.is_error()
            False
        """
        return self.severity.is_error()

    def is_blocker(self) -> bool:
        """
        Check if this issue should block operation.

        Returns:
            bool: True if this issue should prevent operation

        Examples:
            >>> issue = ValidationIssue(ValidationSeverity.FATAL, "Security violation")
            >>> issue.is_blocker()
            True
            >>> warning = ValidationIssue(ValidationSeverity.ERROR, "Data inconsistency")
            >>> warning.is_blocker()
            False
        """
        return self.severity.is_blocker()

    def __str__(self) -> str:
        """
        String representation of the validation issue.

        Returns:
            str: Formatted description of the issue

        Examples:
            >>> str(ValidationIssue(ValidationSeverity.ERROR, "Invalid data"))
            'ERROR: Invalid data'
            >>> str(ValidationIssue(ValidationSeverity.WARNING, "Unusual value", path="config.timeout"))
            'WARNING at config.timeout: Unusual value'
        """
        if self.path:
            return f"{self.severity.name} at {self.path}: {self.message}"
        return f"{self.severity.name}: {self.message}"


class ValidationReport:
    """
    Comprehensive report of validation results including all issues found.

    This class collects and organizes validation issues, providing methods
    to query and analyze validation results in detail.

    Attributes:
        issues (List[ValidationIssue]): List of all validation issues found

    Examples:
        >>> report = ValidationReport()
        >>> report.add_error("Invalid email")
        >>> report.add_warning("Name unusually short", path="user.name")
        >>> report.is_valid()
        False
        >>> report.has_warnings()
        True
        >>> len(report.get_issues())
        2
    """

    def __init__(self) -> None:
        """
        Initialize an empty ValidationReport.
        """
        self.issues: List[ValidationIssue] = []

    def add_issue(self, issue: ValidationIssue) -> None:
        """
        Add a validation issue to the report.

        Args:
            issue (ValidationIssue): The validation issue to add

        Examples:
            >>> report = ValidationReport()
            >>> report.add_issue(ValidationIssue(ValidationSeverity.ERROR, "Invalid data"))
            >>> len(report.issues)
            1
        """
        self.issues.append(issue)

    def add_error(
        self,
        message: str,
        path: Optional[str] = None,
        context: Optional[Dict[str, object]] = None,
    ) -> None:
        """
        Add an error issue to the report.

        Args:
            message (str): Description of the error
            path (Optional[str], optional): Path to the location of the error. Defaults to None.
            context (Optional[Dict[str, object]], optional): Additional contextual information. Defaults to None.

        Examples:
            >>> report = ValidationReport()
            >>> report.add_error("Invalid email format", path="user.email")
            >>> report.issues[0].severity
            <ValidationSeverity.ERROR: 'error'>
        """
        self.add_issue(
            ValidationIssue(ValidationSeverity.ERROR, message, path, context),
        )

    def add_warning(
        self,
        message: str,
        path: Optional[str] = None,
        context: Optional[Dict[str, object]] = None,
    ) -> None:
        """
        Add a warning issue to the report.

        Args:
            message (str): Description of the warning
            path (Optional[str], optional): Path to the location of the warning. Defaults to None.
            context (Optional[Dict[str, object]], optional): Additional contextual information. Defaults to None.

        Examples:
            >>> report = ValidationReport()
            >>> report.add_warning("Unusual value", path="settings.timeout")
            >>> report.issues[0].severity
            <ValidationSeverity.WARNING: 'warning'>
        """
        self.add_issue(
            ValidationIssue(ValidationSeverity.WARNING, message, path, context),
        )

    def add_info(
        self,
        message: str,
        path: Optional[str] = None,
        context: Optional[Dict[str, object]] = None,
    ) -> None:
        """
        Add an informational issue to the report.

        Args:
            message (str): Informational message
            path (Optional[str], optional): Path related to the information. Defaults to None.
            context (Optional[Dict[str, object]], optional): Additional contextual information. Defaults to None.

        Examples:
            >>> report = ValidationReport()
            >>> report.add_info("Using default value", path="config.timeout")
            >>> report.issues[0].severity
            <ValidationSeverity.INFO: 'info'>
        """
        self.add_issue(ValidationIssue(ValidationSeverity.INFO, message, path, context))

    def is_valid(self) -> bool:
        """
        Check if the validation passed with no errors.

        Returns:
            bool: True if no errors were found, False otherwise

        Examples:
            >>> report = ValidationReport()
            >>> report.is_valid()
            True
            >>> report.add_warning("Minor issue")
            >>> report.is_valid()  # Warnings don't invalidate
            True
            >>> report.add_error("Serious problem")
            >>> report.is_valid()
            False
        """
        return not any(issue.is_error() for issue in self.issues)

    def can_proceed(self) -> bool:
        """
        Check if operation can proceed despite validation issues.

        Returns:
            bool: True if there are no blocking issues, False otherwise

        Examples:
            >>> report = ValidationReport()
            >>> report.add_error("Non-fatal issue")
            >>> report.can_proceed()  # Regular errors don't block
            True
            >>> report.add_issue(ValidationIssue(ValidationSeverity.FATAL, "Security violation"))
            >>> report.can_proceed()
            False
        """
        return not any(issue.is_blocker() for issue in self.issues)

    def has_warnings(self) -> bool:
        """
        Check if the report contains any warnings.

        Returns:
            bool: True if warnings were found, False otherwise

        Examples:
            >>> report = ValidationReport()
            >>> report.has_warnings()
            False
            >>> report.add_warning("Potential issue")
            >>> report.has_warnings()
            True
        """
        return any(
            issue.severity == ValidationSeverity.WARNING for issue in self.issues
        )

    def get_issues(
        self,
        severity: Optional[ValidationSeverity] = None,
    ) -> List[ValidationIssue]:
        """
        Get validation issues, optionally filtered by severity.

        Args:
            severity (Optional[ValidationSeverity], optional): If provided, only return issues of this severity.
                Defaults to None.

        Returns:
            List[ValidationIssue]: List of matching issues

        Examples:
            >>> report = ValidationReport()
            >>> report.add_error("Error 1")
            >>> report.add_warning("Warning 1")
            >>> report.add_info("Info message")
            >>> len(report.get_issues())
            3
            >>> len(report.get_issues(ValidationSeverity.ERROR))
            1
        """
        if severity is None:
            return self.issues.copy()
        return [issue for issue in self.issues if issue.severity == severity]

    def get_errors(self) -> List[ValidationIssue]:
        """
        Get all error issues (ERROR and FATAL).

        Returns:
            List[ValidationIssue]: List of error issues

        Examples:
            >>> report = ValidationReport()
            >>> report.add_error("Error 1")
            >>> report.add_issue(ValidationIssue(ValidationSeverity.FATAL, "Fatal error"))
            >>> report.add_warning("Warning 1")
            >>> len(report.get_errors())
            2
        """
        return [issue for issue in self.issues if issue.is_error()]

    def get_warnings(self) -> List[ValidationIssue]:
        """
        Get all warning issues.

        Returns:
            List[ValidationIssue]: List of warning issues

        Examples:
            >>> report = ValidationReport()
            >>> report.add_warning("Warning 1")
            >>> report.add_warning("Warning 2")
            >>> report.add_error("Error 1")
            >>> len(report.get_warnings())
            2
        """
        return self.get_issues(ValidationSeverity.WARNING)

    def __bool__(self) -> bool:
        """
        Boolean evaluation of validation success.

        Returns:
            bool: True if validation passed (no errors), False otherwise

        Examples:
            >>> report = ValidationReport()
            >>> bool(report)
            True
            >>> report.add_error("Problem found")
            >>> bool(report)
            False
        """
        return self.is_valid()

    def __str__(self) -> str:
        """
        String representation of the validation report.

        Returns:
            str: Summary of validation results

        Examples:
            >>> report = ValidationReport()
            >>> str(report)
            'Validation passed with 0 issues'
            >>> report.add_error("Problem 1")
            >>> report.add_warning("Minor issue")
            >>> str(report)
            'Validation failed with 2 issues (1 errors, 1 warnings)'
        """
        if not self.issues:
            return "Validation passed with 0 issues"

        error_count = len([i for i in self.issues if i.is_error()])
        warning_count = len(
            [i for i in self.issues if i.severity == ValidationSeverity.WARNING],
        )
        info_count = len(self.issues) - error_count - warning_count

        status = (
            "failed"
            if error_count > 0
            else "passed with warnings" if warning_count > 0 else "passed"
        )
        details: List[str] = []
        if error_count > 0:
            details.append(f"{error_count} errors")
        if warning_count > 0:
            details.append(f"{warning_count} warnings")
        if info_count > 0:
            details.append(f"{info_count} info")

        return (
            f"Validation {status} with {len(self.issues)} issues ({', '.join(details)})"
        )


# ──────────────────────────────────────────────────────────────
# Type Validation Functions
# ──────────────────────────────────────────────────────────────


def is_non_empty_string(value: object) -> bool:
    """
    Verify if a value is a non-empty string.

    Performs type checking and emptiness validation in a single operation
    with maximum efficiency.

    Args:
        value: The value to validate. Can be any Python object.

    Returns:
        bool: True if the value is a non-empty string, False otherwise.

    Examples:
        >>> is_non_empty_string("hello")
        True
        >>> is_non_empty_string("")
        False
        >>> is_non_empty_string(123)
        False
        >>> is_non_empty_string(None)
        False

    Note:
        This function uses isinstance for type checking rather than type()
        to properly handle inheritance relationships.
    """
    return isinstance(value, str) and bool(value)


def is_valid_identifier(name: str) -> bool:
    """
    Check if a string is a valid Python identifier.

    Validates that a string can be used as a Python variable,
    function, or class name according to Python syntax rules.

    Args:
        name: The string to check

    Returns:
        bool: True if the string is a valid Python identifier, False otherwise

    Examples:
        >>> is_valid_identifier("valid_name")
        True
        >>> is_valid_identifier("123invalid")
        False
        >>> is_valid_identifier("also-invalid")
        False
        >>> is_valid_identifier("")
        False
        >>> is_valid_identifier("_private")
        True

    Note:
        A valid identifier starts with a letter or underscore and contains
        only letters, numbers, and underscores.
    """
    if not name:
        return False

    # First character must be a letter or underscore
    if not (name[0].isalpha() or name[0] == "_"):
        return False

    # Remaining characters must be alphanumeric or underscore
    for char in name[1:]:
        if not (char.isalnum() or char == "_"):
            return False

    return True


def is_subclass_safe(cls: object, parent: ParentSpecType[object]) -> bool:
    """
    Safely check if a class is a subclass of another class.

    Performs an issubclass check that won't raise TypeError if the first
    argument is not a class, unlike the built-in issubclass function.

    Args:
        cls: The potential subclass to check
        parent: The parent class(es) to check against

    Returns:
        bool: True if cls is a subclass of parent, False otherwise

    Examples:
        >>> is_subclass_safe(str, object)
        True
        >>> is_subclass_safe("not_a_class", object)
        False
        >>> is_subclass_safe(dict, (list, tuple))
        False
        >>> is_subclass_safe(list, (list, tuple))
        True
        >>> is_subclass_safe(None, object)
        False

    Note:
        This is a safer version of the built-in issubclass() function that
        won't raise TypeError for non-class objects.
    """
    if not isinstance(cls, type):
        return False

    try:
        # Helper function to check if a type is Callable
        def is_callable_type(typ: object) -> bool:
            return typ is Callable or get_origin(typ) is Callable

        # Handle tuple of types case
        if isinstance(parent, tuple):
            # Check for Callable in parent tuple
            has_callable_type = False
            for p in parent:
                if is_callable_type(p):
                    has_callable_type = True
                    break

            if has_callable_type and callable(cls):
                return True

            return issubclass(cls, parent)

        # Handle single type case
        # Special handling for Callable
        if is_callable_type(parent):
            return callable(cls)

        return issubclass(cls, parent)
    except TypeError:
        return False


def is_instance_of_any(value: object, types: Tuple[Type[object], ...]) -> bool:
    """
    Check if a value is an instance of any of the specified types.

    Determines whether the value is an instance of at least one
    of the types in the provided tuple.

    Args:
        value: The value to check
        types: Tuple of types to check against

    Returns:
        bool: True if value is an instance of any type in types, False otherwise

    Examples:
        >>> is_instance_of_any(42, (str, int, float))
        True
        >>> is_instance_of_any("hello", (list, tuple, dict))
        False
        >>> is_instance_of_any(None, (str, int, type(None)))
        True
        >>> is_instance_of_any([], (list, tuple))
        True

    Note:
        More efficient than multiple isinstance() calls when checking
        against many types.
    """
    return isinstance(value, types)


def is_protocol_instance(obj: object, protocol: Type[object]) -> bool:
    """
    Check if an object satisfies a Protocol interface.

    Safely determines if an object implements all the methods and attributes
    required by a Protocol, with proper handling of runtime Protocol checking.

    Args:
        obj: Object to check
        protocol: Protocol to check against

    Returns:
        bool: True if the object satisfies the protocol, False otherwise

    Examples:
        >>> from typing import Protocol
        >>> class SupportsLen(Protocol):
        ...     def __len__(self) -> int: ...
        >>> is_protocol_instance([1, 2, 3], SupportsLen)  # doctest: +SKIP
        True
        >>> is_protocol_instance(42, SupportsLen)  # doctest: +SKIP
        False

    Note:
        Works with both @runtime_checkable Protocols and regular Protocols.
        For non-runtime-checkable protocols, uses attribute inspection.
    """
    # First try runtime protocol checking if available
    try:
        if hasattr(protocol, "_is_runtime_protocol") and getattr(
            protocol,
            "_is_runtime_protocol",
            False,
        ):
            return isinstance(obj, protocol)  # type: ignore
    except (TypeError, AttributeError):
        pass

    # Fall back to manual attribute checking for non-runtime protocols
    if not hasattr(protocol, "__annotations__"):
        return False

    # Check if object has all required attributes and methods
    return all(hasattr(obj, attr_name) for attr_name in protocol.__annotations__)


def is_numeric(value: object) -> bool:
    """
    Check if a value is numeric (int, float, complex, or numeric subclass).

    Determines whether a value is of a numeric type, handling both
    built-in numeric types and numbers.Number subclasses.

    Args:
        value: The value to check

    Returns:
        bool: True if the value is numeric, False otherwise

    Examples:
        >>> is_numeric(42)
        True
        >>> is_numeric(3.14)
        True
        >>> is_numeric(1+2j)
        True
        >>> is_numeric("42")
        False
        >>> is_numeric(None)
        False

    Note:
        This function considers all subclasses of numbers.Number as numeric.
    """
    return isinstance(value, numbers.Number)


def is_collection(value: object) -> bool:
    """
    Check if a value is a collection (list, tuple, set, dict, etc.).

    Determines whether a value is a collection type that can contain
    multiple elements, excluding strings and bytes which are sequence
    types but not typically treated as collections.

    Args:
        value: The value to check

    Returns:
        bool: True if the value is a collection, False otherwise

    Examples:
        >>> is_collection([1, 2, 3])
        True
        >>> is_collection((1, 2, 3))
        True
        >>> is_collection({"a": 1})
        True
        >>> is_collection("string")  # Strings are not considered collections
        False
        >>> is_collection(42)
        False

    Note:
        Strings and bytes are not considered collections despite being sequences.
    """
    # Check if iterable but exclude strings and bytes
    return (
        hasattr(value, "__iter__")
        and not isinstance(value, (str, bytes))
        and value is not None
    )


def is_callable(value: object) -> bool:
    """
    Check if a value is callable (function, method, callable object).

    Determines whether an object can be called like a function.

    Args:
        value: The value to check

    Returns:
        bool: True if the value is callable, False otherwise

    Examples:
        >>> is_callable(lambda x: x)
        True
        >>> is_callable(print)
        True
        >>> is_callable("not_callable")
        False
        >>> is_callable(None)
        False

    Note:
        This is a type-safe wrapper around the built-in callable() function.
    """
    return callable(value)


def has_attributes(obj: object, *attributes: str) -> bool:
    """
    Check if an object has all the specified attributes.

    Args:
        obj: The object to check
        *attributes: Attribute names to look for

    Returns:
        bool: True if object has all attributes, False otherwise

    Examples:
        >>> has_attributes([], "append", "extend")
        True
        >>> has_attributes({}, "update", "missing_attr")
        False
        >>> has_attributes("string", "upper", "lower")
        True
        >>> has_attributes(None, "any_attr")
        False

    Note:
        This checks for attribute existence, not their values or callability.
    """
    if obj is None:
        return False

    return all(hasattr(obj, attr) for attr in attributes)


def is_method(obj: object) -> bool:
    """
    Check if an object is a method.

    Determines whether an object is a method bound to a class instance.

    Args:
        obj: The object to check

    Returns:
        bool: True if the object is a method, False otherwise

    Examples:
        >>> class Example:
        ...     def method(self): pass
        ...     @classmethod
        ...     def class_method(cls): pass
        ...     @staticmethod
        ...     def static_method(): pass
        >>> obj = Example()
        >>> is_method(obj.method)  # doctest: +SKIP
        True
        >>> is_method(Example.method)  # doctest: +SKIP
        False
        >>> is_method(obj.class_method)  # doctest: +SKIP
        True
        >>> is_method(obj.static_method)  # doctest: +SKIP
        False

    Note:
        This distinguishes between bound methods and regular functions.
    """
    return inspect.ismethod(obj)


def is_function(obj: object) -> bool:
    """
    Check if an object is a function.

    Determines whether an object is a function (not a method or builtin).

    Args:
        obj: The object to check

    Returns:
        bool: True if the object is a function, False otherwise

    Examples:
        >>> def example_function(): pass
        >>> is_function(example_function)
        True
        >>> is_function(len)
        False
        >>> class Example:
        ...     def method(self): pass
        >>> obj = Example()
        >>> is_function(obj.method)  # doctest: +SKIP
        False
        >>> is_function(lambda x: x)
        True

    Note:
        This identifies pure functions, not methods or built-in functions.
    """
    return inspect.isfunction(obj)


def is_compatible_with_type(value: object, target_type: Type[T]) -> bool:
    """
    Check if a value can be converted to a target type without errors.

    Determines whether a value can be safely converted to the specified type
    without raising exceptions, allowing for type conversion safety checks.

    Args:
        value: The value to check
        target_type: The target type to check compatibility with

    Returns:
        bool: True if the value can be safely converted, False otherwise

    Examples:
        >>> is_compatible_with_type("123", int)
        True
        >>> is_compatible_with_type("hello", int)
        False
        >>> is_compatible_with_type(42, str)
        True
        >>> is_compatible_with_type([1, 2, 3], tuple)
        True

    Note:
        This performs actual conversion attempts and catches exceptions,
        making it suitable for runtime type compatibility checking.
    """
    # Already the correct type
    if isinstance(value, target_type):
        return True

    # Handle special conversions
    try:
        if target_type is int:
            if isinstance(value, (str, float, bool, bytes, SupportsInt)):
                int(value)  # type: ignore
                return True
            return False

        if target_type is float:
            if isinstance(value, (str, int, bool, SupportsFloat)):
                float(value)  # type: ignore
                return True
            return False

        if target_type is bool:
            # Everything can be converted to bool
            return True

        if target_type is str:
            # Everything can be converted to str
            return True

        if target_type is bytes and isinstance(value, str):
            bytes(value, "utf-8")
            return True

        if target_type is list and is_collection(value):
            # Safe check for iterables
            if isinstance(value, Iterable):
                list(value)  # type: ignore
                return True
            return False

        if target_type is tuple and is_collection(value):
            if isinstance(value, Iterable):
                tuple(value)  # type: ignore
                return True
            return False

        if target_type is set and is_collection(value):
            if isinstance(value, Iterable):
                set(value)  # type: ignore
                return True
            return False

        if target_type is dict and hasattr(value, "items"):
            dict(value)  # type: ignore
            return True

        # Try direct conversion as a last resort
        # This may raise an exception
        if isinstance(value, Iterable) and issubclass(target_type, Collection):
            target_type(value)  # type: ignore
            return True

        return False
    except (ValueError, TypeError):
        return False


def are_types_compatible(source_type: Type[object], target_type: Type[object]) -> bool:
    """
    Check if two types are compatible for conversion or assignment.

    Determines whether values of the source type can generally be
    converted to the target type without errors.

    Args:
        source_type: The source type to check from
        target_type: The target type to check compatibility with

    Returns:
        bool: True if the types are compatible, False otherwise

    Examples:
        >>> are_types_compatible(int, float)
        True
        >>> are_types_compatible(float, int)
        True
        >>> are_types_compatible(list, tuple)
        True
        >>> are_types_compatible(dict, list)
        False

    Note:
        This evaluates type compatibility at a general level without
        considering specific value constraints.
    """
    # Same types are always compatible
    if source_type is target_type:
        return True

    # Special numeric type compatibility
    if is_subclass_safe(source_type, numbers.Number) and is_subclass_safe(
        target_type,
        numbers.Number,
    ):
        # All numeric types are generally compatible with each other
        return True

    # Collection type compatibility
    if is_subclass_safe(source_type, Collection) and is_subclass_safe(
        target_type,
        Collection,
    ):
        # Check for specific collection type compatibility
        if (
            (
                is_subclass_safe(source_type, Sequence)
                and is_subclass_safe(target_type, Sequence)
            )
            or (
                is_subclass_safe(source_type, Set)
                and is_subclass_safe(target_type, Set)
            )
            or (
                is_subclass_safe(source_type, Mapping)
                and is_subclass_safe(target_type, Mapping)
            )
        ):
            return True

    # String types compatibility
    if source_type is str and target_type is bytes:
        return True  # str can be converted to bytes

    # Check for general conversion compatibility
    try:
        # Use Union type to properly type the sample value for all possible types
        sample_value: Optional[
            Union[
                int,
                float,
                bool,
                str,
                bytes,
                List[Any],
                Tuple[Any, ...],
                Dict[Any, Any],
                Set[Any],
            ]
        ] = None

        # Try to create a minimal sample value for the source type
        if source_type is int:
            sample_value = 0
        elif source_type is float:
            sample_value = 0.0
        elif source_type is bool:
            sample_value = False
        elif source_type is str:
            sample_value = ""
        elif source_type is bytes:
            sample_value = b""
        elif source_type is list:
            sample_value = []
        elif source_type is tuple:
            sample_value = ()
        elif source_type is dict:
            sample_value = {}
        elif source_type is set:
            sample_value = set()

        if sample_value is not None:
            return is_compatible_with_type(sample_value, target_type)
    except (ValueError, TypeError):
        return False

    # Default case when no specific compatibility is found
    return False
