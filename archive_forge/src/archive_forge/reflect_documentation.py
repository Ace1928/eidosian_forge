import os
import pickle
import re
import sys
import traceback
import types
import weakref
from collections import deque
from io import IOBase, StringIO
from typing import Type, Union
from twisted.python.compat import nativeString
from twisted.python.deprecate import _fullyQualifiedName as fullyQualifiedName

    L{objgrep} finds paths between C{start} and C{goal}.

    Starting at the python object C{start}, we will loop over every reachable
    reference, tring to find the python object C{goal} (i.e. every object
    C{candidate} for whom C{eq(candidate, goal)} is truthy), and return a
    L{list} of L{str}, where each L{str} is Python syntax for a path between
    C{start} and C{goal}.

    Since this can be slightly difficult to visualize, here's an example::

        >>> class Holder:
        ...     def __init__(self, x):
        ...         self.x = x
        ...
        >>> start = Holder({"irrelevant": "ignore",
        ...                 "relevant": [7, 1, 3, 5, 7]})
        >>> for path in objgrep(start, 7):
        ...     print("start" + path)
        start.x['relevant'][0]
        start.x['relevant'][4]

    This can be useful, for example, when debugging stateful graphs of objects
    attached to a socket, trying to figure out where a particular connection is
    attached.

    @param start: The object to start looking at.

    @param goal: The object to search for.

    @param eq: A 2-argument predicate which takes an object found by traversing
        references starting at C{start}, as well as C{goal}, and returns a
        boolean.

    @param path: The prefix of the path to include in every return value; empty
        by default.

    @param paths: The result object to append values to; a list of strings.

    @param seen: A dictionary mapping ints (object IDs) to objects already
        seen.

    @param showUnknowns: if true, print a message to C{stdout} when
        encountering objects that C{objgrep} does not know how to traverse.

    @param maxDepth: The maximum number of object references to attempt
        traversing before giving up.  If an integer, limit to that many links,
        if C{None}, unlimited.

    @return: A list of strings representing python object paths starting at
        C{start} and terminating at C{goal}.
    