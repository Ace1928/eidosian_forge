import typing
from typing import Awaitable, Callable, Any, List, Union
import asyncio

# Define a type alias for a validation rule function
ValidationRule = Callable[[Any], Awaitable[bool]]


class IntegerValidators:
    """
    This class encapsulates all validators for integer values, ensuring each validator adheres to the specified
    interface of accepting a single value of any type and returning an Awaitable that resolves to a boolean.
    """

    @staticmethod
    async def is_integer(value: Any) -> bool:
        """
        Validates if the provided value is of integer type.

        Args:
            value (Any): The value to validate.

        Returns:
            bool: True if the value is an integer, False otherwise.
        """
        return isinstance(value, int)

    @staticmethod
    async def is_positive(value: Any) -> bool:
        """
        Validates if the provided integer value is positive.

        Args:
            value (Any): The value to validate.

        Returns:
            bool: True if the value is a positive integer, False otherwise.
        """
        return await IntegerValidators.is_integer(value) and value > 0

    @staticmethod
    async def is_negative(value: Any) -> bool:
        """
        Validates if the provided integer value is negative.

        Args:
            value (Any): The value to validate.

        Returns:
            bool: True if the value is a negative integer, False otherwise.
        """
        return await IntegerValidators.is_integer(value) and value < 0

    @staticmethod
    async def is_even(value: Any) -> bool:
        """
        Validates if the provided integer value is even.

        Args:
            value (Any): The value to validate.

        Returns:
            bool: True if the value is an even integer, False otherwise.
        """
        return await IntegerValidators.is_integer(value) and value % 2 == 0

    @staticmethod
    async def is_odd(value: Any) -> bool:
        """
        Validates if the provided integer value is odd.

        Args:
            value (Any): The value to validate.

        Returns:
            bool: True if the value is an odd integer, False otherwise.
        """
        return await IntegerValidators.is_integer(value) and value % 2 != 0

    @staticmethod
    async def is_multiple_of(value: Any, multiple: int) -> bool:
        """
        Validates if the provided integer value is a multiple of another specified integer.

        Args:
            value (Any): The value to validate.
            multiple (int): The integer to check multiplicity against.

        Returns:
            bool: True if the value is a multiple of the specified integer, False otherwise.
        """
        return await IntegerValidators.is_integer(value) and value % multiple == 0

    @staticmethod
    async def is_between(value: Any, lower_bound: int, upper_bound: int) -> bool:
        """
        Validates if the provided integer value is between two specified integers (inclusive).

        Args:
            value (Any): The value to validate.
            lower_bound (int): The lower bound of the range.
            upper_bound (int): The upper bound of the range.

        Returns:
            bool: True if the value is within the specified range, False otherwise.
        """
        return (
            await IntegerValidators.is_integer(value)
            and lower_bound <= value <= upper_bound
        )

    @staticmethod
    async def is_in(value: Any, values: List[int]) -> bool:
        """
        Validates if the provided integer value is contained within a specified list of integers.

        Args:
            value (Any): The value to validate.
            values (List[int]): The list of integers to check against.

        Returns:
            bool: True if the value is contained within the list, False otherwise.
        """
        return await IntegerValidators.is_integer(value) and value in values

    @staticmethod
    async def is_not_in(value: Any, values: List[int]) -> bool:
        """
        Validates if the provided integer value is not contained within a specified list of integers.

        Args:
            value (Any): The value to validate.
            values (List[int]): The list of integers to check against.

        Returns:
            bool: True if the value is not contained within the list, False otherwise.
        """
        return await IntegerValidators.is_integer(value) and value not in values

    @staticmethod
    async def is_positive_multiple_of(value: Any, multiple: int) -> bool:
        """
        Validates if the provided integer value is positive and a multiple of another specified integer.

        Args:
            value (Any): The value to validate.
            multiple (int): The integer to check multiplicity against.

        Returns:
            bool: True if the value is positive and a multiple of the specified integer, False otherwise.
        """
        return await IntegerValidators.is_positive(value) and value % multiple == 0

    @staticmethod
    async def is_negative_multiple_of(value: Any, multiple: int) -> bool:
        """
        Validates if the provided integer value is negative and a multiple of another specified integer.

        Args:
            value (Any): The value to validate.
            multiple (int): The integer to check multiplicity against.

        Returns:
            bool: True if the value is negative and a multiple of the specified integer, False otherwise.
        """
        return await IntegerValidators.is_negative(value) and value % multiple == 0

    @staticmethod
    async def is_positive_between(
        value: Any, lower_bound: int, upper_bound: int
    ) -> bool:
        """
        Validates if the provided integer value is positive and between two specified integers (exclusive).

        Args:
            value (Any): The value to validate.
            lower_bound (int): The lower bound of the range.
            upper_bound (int): The upper bound of the range.

        Returns:
            bool: True if the value is positive and within the specified range, False otherwise.
        """
        return (
            await IntegerValidators.is_positive(value)
            and lower_bound < value < upper_bound
        )

    @staticmethod
    async def is_negative_between(
        value: Any, lower_bound: int, upper_bound: int
    ) -> bool:
        """
        Validates if the provided integer value is negative and between two specified integers (exclusive).

        Args:
            value (Any): The value to validate.
            lower_bound (int): The lower bound of the range.
            upper_bound (int): The upper bound of the range.

        Returns:
            bool: True if the value is negative and within the specified range, False otherwise.
        """
        return (
            await IntegerValidators.is_negative(value)
            and lower_bound < value < upper_bound
        )

    @staticmethod
    async def is_positive_in(value: Any, values: List[int]) -> bool:
        """
        Validates if the provided integer value is positive and contained within a specified list of integers.

        Args:
            value (Any): The value to validate.
            values (List[int]): The list of integers to check against.

        Returns:
            bool: True if the value is positive and contained within the list, False otherwise.
        """
        return await IntegerValidators.is_positive(value) and value in values

    @staticmethod
    async def is_negative_in(value: Any, values: List[int]) -> bool:
        """
        Validates if the provided integer value is negative and contained within a specified list of integers.

        Args:
            value (Any): The value to validate.
            values (List[int]): The list of integers to check against.

        Returns:
            bool: True if the value is negative and contained within the list, False otherwise.
        """
        return await IntegerValidators.is_negative(value) and value in values

    @staticmethod
    async def is_positive_not_in(value: Any, values: List[int]) -> bool:
        """
        Validates if the provided integer value is positive and not contained within a specified list of integers.

        Args:
            value (Any): The value to validate.
            values (List[int]): The list of integers to check against.

        Returns:
            bool: True if the value is positive and not contained within the list, False otherwise.
        """
        return await IntegerValidators.is_positive(value) and value not in values
