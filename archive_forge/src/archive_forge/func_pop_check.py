import abc
import ast
import inspect
import stevedore
def pop_check(self):
    """Pops the last check from the list and returns them

        :returns: self, the popped check
        :rtype: :class:`.OrCheck`, class:`.Check`
        """
    check = self.rules.pop()
    return (self, check)