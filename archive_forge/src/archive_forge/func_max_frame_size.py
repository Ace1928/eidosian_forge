import collections
import enum
from hyperframe.frame import SettingsFrame
from h2.errors import ErrorCodes
from h2.exceptions import InvalidSettingsValueError
@max_frame_size.setter
def max_frame_size(self, value):
    self[SettingCodes.MAX_FRAME_SIZE] = value