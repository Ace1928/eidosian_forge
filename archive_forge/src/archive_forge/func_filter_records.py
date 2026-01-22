from __future__ import (absolute_import, division, print_function)
import abc
from ansible.module_utils import six
from ansible_collections.community.dns.plugins.module_utils.zone import (
def filter_records(records, prefix=NOT_PROVIDED, record_type=NOT_PROVIDED):
    """
    Given a list of records, returns a filtered subset.

    @param prefix: The prefix to filter for, if provided. Since None is a valid value,
                   the special constant NOT_PROVIDED indicates that we are not filtering.
    @param record_type: The record type to filter for, if provided
    @return The list of records matching the provided filters.
    """
    if prefix is not NOT_PROVIDED:
        records = [record for record in records if record.prefix == prefix]
    if record_type is not NOT_PROVIDED:
        records = [record for record in records if record.type == record_type]
    return records