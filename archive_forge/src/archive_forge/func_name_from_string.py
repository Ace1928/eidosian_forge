import datetime
import struct
from six.moves import winreg
from six import text_type
from ._common import tzrangebase
def name_from_string(self, tzname_str):
    """
        Parse strings as returned from the Windows registry into the time zone
        name as defined in the registry.

        >>> from dateutil.tzwin import tzres
        >>> tzr = tzres()
        >>> print(tzr.name_from_string('@tzres.dll,-251'))
        'Dateline Daylight Time'
        >>> print(tzr.name_from_string('Eastern Standard Time'))
        'Eastern Standard Time'

        :param tzname_str:
            A timezone name string as returned from a Windows registry key.

        :return:
            Returns the localized timezone string from tzres.dll if the string
            is of the form `@tzres.dll,-offset`, else returns the input string.
        """
    if not tzname_str.startswith('@'):
        return tzname_str
    name_splt = tzname_str.split(',-')
    try:
        offset = int(name_splt[1])
    except:
        raise ValueError('Malformed timezone string.')
    return self.load_name(offset)