import operator
from django.utils.hashable import make_hashable
@property
def message_dict(self):
    getattr(self, 'error_dict')
    return dict(self)