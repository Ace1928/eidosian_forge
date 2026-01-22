from __future__ import annotations
import os
import re
from monty.io import zopen
def micro_pyawk(filename, search, results=None, debug=None, postdebug=None):
    """Small awk-mimicking search routine.

    'file' is file to search through.
    'search' is the "search program", a list of lists/tuples with 3 elements;
    i.e. [[regex,test,run],[regex,test,run],...]
    'results' is a an object that your search program will have access to for
    storing results.

    Here regex is either as a Regex object, or a string that we compile into a
    Regex. test and run are callable objects.

    This function goes through each line in filename, and if regex matches that
    line *and* test(results,line)==True (or test is None) we execute
    run(results,match),where match is the match object from running
    Regex.match.

    The default results is an empty dictionary. Passing a results object let
    you interact with it in run() and test(). Hence, in many occasions it is
    thus clever to use results=self.

    Author: Rickard Armiento, Ioannis Petousis

    Returns:
        results
    """
    if results is None:
        results = {}
    for entry in search:
        entry[0] = re.compile(entry[0])
    with zopen(filename, mode='rt') as file:
        for line in file:
            for entry in search:
                match = re.search(entry[0], line)
                if match and (entry[1] is None or entry[1](results, line)):
                    if debug is not None:
                        debug(results, match)
                    entry[2](results, match)
                    if postdebug is not None:
                        postdebug(results, match)
    return results