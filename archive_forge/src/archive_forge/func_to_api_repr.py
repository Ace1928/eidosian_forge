import collections
import collections.abc
import operator
import warnings
def to_api_repr(self):
    """Render a JSON policy resource.

        Returns:
            dict: a resource to be passed to the ``setIamPolicy`` API.
        """
    resource = {}
    if self.etag is not None:
        resource['etag'] = self.etag
    if self.version is not None:
        resource['version'] = self.version
    if self._bindings and len(self._bindings) > 0:
        bindings = []
        for binding in self._bindings:
            members = binding.get('members')
            if members:
                new_binding = {'role': binding['role'], 'members': sorted(members)}
                condition = binding.get('condition')
                if condition:
                    new_binding['condition'] = condition
                bindings.append(new_binding)
        if bindings:
            key = operator.itemgetter('role')
            resource['bindings'] = sorted(bindings, key=key)
    return resource