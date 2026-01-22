from __future__ import absolute_import, division, print_function
from ansible_collections.community.routeros.plugins.module_utils.version import LooseVersion
def specialize_for_version(self, api_version):
    fields = self.fields.copy()
    for conditions, name, field in self.versioned_fields:
        matching = True
        for other_version, comparator in conditions:
            other_api_version = LooseVersion(other_version)
            if not _compare(api_version, other_api_version, comparator):
                matching = False
                break
        if matching:
            if name in fields:
                raise ValueError('Internal error: field "{field}" already exists for {version}'.format(field=name, version=api_version))
            fields[name] = field
    return VersionedAPIData(primary_keys=self.primary_keys, stratify_keys=self.stratify_keys, required_one_of=self.required_one_of, mutually_exclusive=self.mutually_exclusive, has_identifier=self.has_identifier, single_value=self.single_value, unknown_mechanism=self.unknown_mechanism, fully_understood=self.fully_understood, fixed_entries=self.fixed_entries, fields=fields)