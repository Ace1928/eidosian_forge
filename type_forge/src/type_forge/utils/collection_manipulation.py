"""
Collection manipulation utilities for the Type Forge system.

This module provides specialized functions for manipulating collections
with type safety and algorithmic efficiency. Each function is crafted to
transform or analyze collections with predictable complexity guarantees.

Key capabilities:
- List deduplication with order preservation
- Dictionary filtering and transformation
- Collection partitioning and grouping
- List flattening and structure manipulation
- Dictionary transposition and remapping

These utilities form a foundation for type operations throughout the
Type Forge system, ensuring consistent handling of collection types
while maintaining strong typing and performance characteristics.
"""

from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

from type_forge.typing.definitions import HashableT, K, T, V

# ──────────────────────────────────────────────────────────────
# Collection Manipulation Functions
# ──────────────────────────────────────────────────────────────


def deduplicate_list(data: List[HashableT]) -> List[HashableT]:
    """
    Remove duplicate elements from a list while preserving original order.

    Uses an OrderedDict for O(n) deduplication efficiency while guaranteeing
    the original order of elements is maintained.

    Args:
        data (List[HashableT]): A list of hashable elements that may contain duplicates.
            All elements must be hashable (support the __hash__ method).

    Returns:
        List[HashableT]: A new list containing unique elements from the input list,
        with the original order preserved.

    Examples:
        >>> deduplicate_list([1, 2, 2, 3, 2])
        [1, 2, 3]
        >>> deduplicate_list(["a", "b", "a", "c"])
        ['a', 'b', 'c']
        >>> deduplicate_list([])
        []

    Note:
        Time complexity: O(n) where n is the length of the input list.
        Space complexity: O(n) for the OrderedDict storage.
    """
    # Use OrderedDict for O(n) deduplication while preserving order
    return list(OrderedDict.fromkeys(data))


def group_by_key(items: List[Dict[K, V]], key: K) -> Dict[V, List[Dict[K, V]]]:
    """
    Group a list of dictionaries by the value of a specified key.

    Creates a dictionary where keys are the distinct values of the specified key
    in the input dictionaries, and values are lists of dictionaries sharing that value.

    Args:
        items (List[Dict[K, V]]): List of dictionaries to group
        key (K): Key to group by

    Returns:
        Dict[V, List[Dict[K, V]]]: Dictionary mapping key values to lists of dictionaries

    Examples:
        >>> people = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}, {"name": "Charlie", "age": 30}]
        >>> grouped = group_by_key(people, "age")
        >>> grouped[30]  # Returns [{"name": "Alice", "age": 30}, {"name": "Charlie", "age": 30}]
        [{'name': 'Alice', 'age': 30}, {'name': 'Charlie', 'age': 30}]
        >>> grouped[25]
        [{'name': 'Bob', 'age': 25}]

    Note:
        Dictionaries without the specified key will be ignored.
        Time complexity: O(n) where n is the number of items.
    """
    result: Dict[V, List[Dict[K, V]]] = {}

    for item in items:
        if key in item:
            value = item[key]
            if value not in result:
                result[value] = []
            result[value].append(item)

    return result


def flatten_list(nested_list: List[List[T]]) -> List[T]:
    """
    Flatten a list of lists into a single list.

    Combines multiple lists into a single list containing all elements
    from the original lists in sequence.

    Args:
        nested_list (List[List[T]]): A list containing other lists to flatten

    Returns:
        List[T]: A single flattened list containing all elements

    Examples:
        >>> flatten_list([[1, 2], [3, 4], [5]])
        [1, 2, 3, 4, 5]
        >>> flatten_list([["a", "b"], ["c"]])
        ['a', 'b', 'c']
        >>> flatten_list([])
        []

    Note:
        Only flattens one level of nesting. For deeper nesting, use recursion.
        Time complexity: O(n) where n is the total number of elements.
    """
    return [item for sublist in nested_list for item in sublist]


def partition_list(
    items: List[T],
    predicate: Callable[[T], bool],
) -> Tuple[List[T], List[T]]:
    """
    Partition a list into two lists based on a predicate function.

    Divides the input list into two lists: one containing elements for which
    the predicate returns True, and one for elements where it returns False.

    Args:
        items (List[T]): List of items to partition
        predicate (Callable[[T], bool]): Function that determines the partition

    Returns:
        Tuple[List[T], List[T]]: Tuple containing (items_where_true, items_where_false)

    Examples:
        >>> partition_list([1, 2, 3, 4, 5], lambda x: x % 2 == 0)
        ([2, 4], [1, 3, 5])
        >>> partition_list(["apple", "banana", "cherry"], lambda s: len(s) > 5)
        (['banana', 'cherry'], ['apple'])
        >>> partition_list([], lambda x: True)
        ([], [])

    Note:
        Time complexity: O(n) where n is the number of items.
        Space complexity: O(n) for storing the partitioned lists.
    """
    true_items: List[T] = []
    false_items: List[T] = []

    for item in items:
        if predicate(item):
            true_items.append(item)
        else:
            false_items.append(item)

    return (true_items, false_items)


def filter_dict(data: Dict[K, V], predicate: Callable[[K, V], bool]) -> Dict[K, V]:
    """
    Filter a dictionary based on a predicate function.

    Creates a new dictionary containing only the key-value pairs that satisfy
    the predicate function.

    Args:
        data (Dict[K, V]): Dictionary to filter
        predicate (Callable[[K, V], bool]): Function that determines whether to include a pair

    Returns:
        Dict[K, V]: Filtered dictionary

    Examples:
        >>> filter_dict({"a": 1, "b": 2, "c": 3}, lambda k, v: v % 2 == 1)
        {'a': 1, 'c': 3}
        >>> filter_dict({"short": "value", "longer": "string"}, lambda k, v: len(k) > 5)
        {'longer': 'string'}
        >>> filter_dict({}, lambda k, v: True)
        {}

    Note:
        Creates a new dictionary rather than modifying the input dictionary.
        Time complexity: O(n) where n is the number of key-value pairs.
    """
    return {k: v for k, v in data.items() if predicate(k, v)}


def transpose_dict_of_lists(data: Dict[K, List[V]]) -> Dict[V, List[K]]:
    """
    Transpose a dictionary of lists.

    Converts a dictionary mapping keys to lists of values into a dictionary
    mapping values to lists of keys that contained them.

    Args:
        data (Dict[K, List[V]]): Dictionary mapping keys to lists of values

    Returns:
        Dict[V, List[K]]: Transposed dictionary mapping values to lists of keys

    Examples:
        >>> transpose_dict_of_lists({"a": [1, 2], "b": [2, 3]})
        {1: ['a'], 2: ['a', 'b'], 3: ['b']}
        >>> transpose_dict_of_lists({"x": []})
        {}
        >>> transpose_dict_of_lists({})
        {}

    Note:
        All values must be hashable for this operation to work.
        Time complexity: O(n) where n is the total number of values across all lists.
    """
    result: Dict[V, List[K]] = {}

    for key, values in data.items():
        for value in values:
            if value not in result:
                result[value] = []
            result[value].append(key)

    return result
