from django.conf import settings
from django.contrib.messages import constants, utils
from django.utils.functional import SimpleLazyObject
@property
def level_tag(self):
    return LEVEL_TAGS.get(self.level, '')