from __future__ import absolute_import, division, print_function
import pytest
from ansible.errors import AnsibleError
from awx.main.models import JobTemplate, Schedule
from awx.api.serializers import SchedulePreviewSerializer
@pytest.mark.parametrize('freq, kwargs, msg', [('minute', {'start_date': '2020-4-16 03:45:07', 'end_on': 'junk'}, 'Parameter end_on must either be an integer or in the format YYYY-MM-DD'), ('week', {'start_date': '2020-4-16 03:45:07', 'on_days': 'junk'}, 'Parameter on_days must only contain values monday, tuesday, wednesday, thursday, friday, saturday, sunday'), ('month', dict(start_date='2020-4-16 03:45:07', on_the='something', month_day_number='else'), 'Month based frequencies can have month_day_number or on_the but not both'), ('month', dict(start_date='2020-4-16 03:45:07', month_day_number='junk'), 'month_day_number must be between 1 and 31'), ('month', dict(start_date='2020-4-16 03:45:07', month_day_number='0'), 'month_day_number must be between 1 and 31'), ('month', dict(start_date='2020-4-16 03:45:07', month_day_number='32'), 'month_day_number must be between 1 and 31'), ('month', dict(start_date='2020-4-16 03:45:07', on_the='junk'), 'on_the parameter must be two words separated by a space'), ('month', dict(start_date='2020-4-16 03:45:07', on_the='junk wednesday'), 'The first string of the on_the parameter is not valid'), ('month', dict(start_date='2020-4-16 03:45:07', on_the='second junk'), 'Weekday portion of on_the parameter is not valid'), ('month', dict(start_date='2020-4-16 03:45:07', timezone='junk'), 'Timezone parameter is not valid')])
def test_rrule_lookup_plugin_failure(collection_import, freq, kwargs, msg):
    LookupModule = collection_import('plugins.lookup.schedule_rrule').LookupModule()
    with pytest.raises(AnsibleError) as e:
        assert LookupModule.get_rrule(freq, kwargs)
    assert msg in str(e.value)