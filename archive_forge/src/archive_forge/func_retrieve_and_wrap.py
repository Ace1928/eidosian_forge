import importlib
import inspect
def retrieve_and_wrap(self, obj):
    related_obj = self.get(obj)
    return self.wrap(related_obj)