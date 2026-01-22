from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils._text import to_native
def profile_to_dict(profile):
    if not profile:
        return dict()
    result = dict(name=to_native(profile.name), count=to_native(profile.capacity.default), max_count=to_native(profile.capacity.maximum), min_count=to_native(profile.capacity.minimum))
    if profile.rules:
        result['rules'] = [rule_to_dict(r) for r in profile.rules]
    if profile.fixed_date:
        result['fixed_date_timezone'] = profile.fixed_date.time_zone
        result['fixed_date_start'] = profile.fixed_date.start
        result['fixed_date_end'] = profile.fixed_date.end
    if profile.recurrence:
        if get_enum_value(profile.recurrence.frequency) != 'None':
            result['recurrence_frequency'] = get_enum_value(profile.recurrence.frequency)
        if profile.recurrence.schedule:
            result['recurrence_timezone'] = to_native(str(profile.recurrence.schedule.time_zone))
            result['recurrence_days'] = [to_native(r) for r in profile.recurrence.schedule.days]
            result['recurrence_hours'] = [to_native(r) for r in profile.recurrence.schedule.hours]
            result['recurrence_mins'] = [to_native(r) for r in profile.recurrence.schedule.minutes]
    return result