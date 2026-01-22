import collections
import copy
import functools
import itertools
import operator
from heat.common import exception
from heat.engine import function
from heat.engine import properties
def required_resource_names(self):
    """Return a set of names of all resources on which this depends.

        Note that this is done entirely in isolation from the rest of the
        template, so the resource names returned may refer to resources that
        don't actually exist, or would have strict_dependency=False. Use the
        dependencies() method to get validated dependencies.
        """
    if self._dep_names is None:
        explicit_depends = [] if self._depends is None else self._depends

        def path(section):
            return '.'.join([self.name, section])
        prop_deps = function.dependencies(self._properties, path(PROPERTIES))
        metadata_deps = function.dependencies(self._metadata, path(METADATA))
        implicit_depends = map(lambda rp: rp.name, itertools.chain(prop_deps, metadata_deps))
        if self.external_id():
            if explicit_depends:
                raise exception.InvalidExternalResourceDependency(external_id=self.external_id(), resource_type=self.resource_type)
            self._dep_names = set()
        else:
            self._dep_names = set(itertools.chain(explicit_depends, implicit_depends))
    return self._dep_names