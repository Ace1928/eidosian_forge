from django.core import checks
def delete_cached_value(self, instance):
    del instance._state.fields_cache[self.get_cache_name()]