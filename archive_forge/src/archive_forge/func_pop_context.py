import types
from .utils import ExactWeakKeyDictionary
def pop_context(self, code: types.CodeType):
    ctx = self.get_context(code)
    self.code_context._remove_id(id(code))
    return ctx