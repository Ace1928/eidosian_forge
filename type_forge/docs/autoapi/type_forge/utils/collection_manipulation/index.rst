type_forge.utils.collection_manipulation
========================================

.. py:module:: type_forge.utils.collection_manipulation

.. autoapi-nested-parse::

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



Functions
---------

.. autoapisummary::

   type_forge.utils.collection_manipulation.deduplicate_list
   type_forge.utils.collection_manipulation.filter_dict
   type_forge.utils.collection_manipulation.flatten_list
   type_forge.utils.collection_manipulation.group_by_key
   type_forge.utils.collection_manipulation.partition_list
   type_forge.utils.collection_manipulation.transpose_dict_of_lists


Module Contents
---------------

.. py:function:: deduplicate_list(data)

   Remove duplicate elements from a list while preserving original order.

   Uses an OrderedDict for O(n) deduplication efficiency while guaranteeing
   the original order of elements is maintained.

   :param data: A list of hashable elements that may contain duplicates.
                All elements must be hashable (support the __hash__ method).
   :type data: List[HashableT]

   :returns: A new list containing unique elements from the input list,
             with the original order preserved.
   :rtype: List[HashableT]

   .. rubric:: Examples

   >>> deduplicate_list([1, 2, 2, 3, 2])
   [1, 2, 3]
   >>> deduplicate_list(["a", "b", "a", "c"])
   ['a', 'b', 'c']
   >>> deduplicate_list([])
   []

   .. note::

      Time complexity: O(n) where n is the length of the input list.
      Space complexity: O(n) for the OrderedDict storage.


.. py:function:: filter_dict(data, predicate)

   Filter a dictionary based on a predicate function.

   Creates a new dictionary containing only the key-value pairs that satisfy
   the predicate function.

   :param data: Dictionary to filter
   :type data: Dict[K, V]
   :param predicate: Function that determines whether to include a pair
   :type predicate: Callable[[K, V], bool]

   :returns: Filtered dictionary
   :rtype: Dict[K, V]

   .. rubric:: Examples

   >>> filter_dict({"a": 1, "b": 2, "c": 3}, lambda k, v: v % 2 == 1)
   {'a': 1, 'c': 3}
   >>> filter_dict({"short": "value", "longer": "string"}, lambda k, v: len(k) > 5)
   {'longer': 'string'}
   >>> filter_dict({}, lambda k, v: True)
   {}

   .. note::

      Creates a new dictionary rather than modifying the input dictionary.
      Time complexity: O(n) where n is the number of key-value pairs.


.. py:function:: flatten_list(nested_list)

   Flatten a list of lists into a single list.

   Combines multiple lists into a single list containing all elements
   from the original lists in sequence.

   :param nested_list: A list containing other lists to flatten
   :type nested_list: List[List[T]]

   :returns: A single flattened list containing all elements
   :rtype: List[T]

   .. rubric:: Examples

   >>> flatten_list([[1, 2], [3, 4], [5]])
   [1, 2, 3, 4, 5]
   >>> flatten_list([["a", "b"], ["c"]])
   ['a', 'b', 'c']
   >>> flatten_list([])
   []

   .. note::

      Only flattens one level of nesting. For deeper nesting, use recursion.
      Time complexity: O(n) where n is the total number of elements.


.. py:function:: group_by_key(items, key)

   Group a list of dictionaries by the value of a specified key.

   Creates a dictionary where keys are the distinct values of the specified key
   in the input dictionaries, and values are lists of dictionaries sharing that value.

   :param items: List of dictionaries to group
   :type items: List[Dict[K, V]]
   :param key: Key to group by
   :type key: K

   :returns: Dictionary mapping key values to lists of dictionaries
   :rtype: Dict[V, List[Dict[K, V]]]

   .. rubric:: Examples

   >>> people = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}, {"name": "Charlie", "age": 30}]
   >>> grouped = group_by_key(people, "age")
   >>> grouped[30]  # Returns [{"name": "Alice", "age": 30}, {"name": "Charlie", "age": 30}]
   [{'name': 'Alice', 'age': 30}, {'name': 'Charlie', 'age': 30}]
   >>> grouped[25]
   [{'name': 'Bob', 'age': 25}]

   .. note::

      Dictionaries without the specified key will be ignored.
      Time complexity: O(n) where n is the number of items.


.. py:function:: partition_list(items, predicate)

   Partition a list into two lists based on a predicate function.

   Divides the input list into two lists: one containing elements for which
   the predicate returns True, and one for elements where it returns False.

   :param items: List of items to partition
   :type items: List[T]
   :param predicate: Function that determines the partition
   :type predicate: Callable[[T], bool]

   :returns: Tuple containing (items_where_true, items_where_false)
   :rtype: Tuple[List[T], List[T]]

   .. rubric:: Examples

   >>> partition_list([1, 2, 3, 4, 5], lambda x: x % 2 == 0)
   ([2, 4], [1, 3, 5])
   >>> partition_list(["apple", "banana", "cherry"], lambda s: len(s) > 5)
   (['banana', 'cherry'], ['apple'])
   >>> partition_list([], lambda x: True)
   ([], [])

   .. note::

      Time complexity: O(n) where n is the number of items.
      Space complexity: O(n) for storing the partitioned lists.


.. py:function:: transpose_dict_of_lists(data)

   Transpose a dictionary of lists.

   Converts a dictionary mapping keys to lists of values into a dictionary
   mapping values to lists of keys that contained them.

   :param data: Dictionary mapping keys to lists of values
   :type data: Dict[K, List[V]]

   :returns: Transposed dictionary mapping values to lists of keys
   :rtype: Dict[V, List[K]]

   .. rubric:: Examples

   >>> transpose_dict_of_lists({"a": [1, 2], "b": [2, 3]})
   {1: ['a'], 2: ['a', 'b'], 3: ['b']}
   >>> transpose_dict_of_lists({"x": []})
   {}
   >>> transpose_dict_of_lists({})
   {}

   .. note::

      All values must be hashable for this operation to work.
      Time complexity: O(n) where n is the total number of values across all lists.


