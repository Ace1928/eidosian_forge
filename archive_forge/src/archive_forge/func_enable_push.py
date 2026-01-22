import collections
import enum
from hyperframe.frame import SettingsFrame
from h2.errors import ErrorCodes
from h2.exceptions import InvalidSettingsValueError
@enable_push.setter
def enable_push(self, value):
    self[SettingCodes.ENABLE_PUSH] = value