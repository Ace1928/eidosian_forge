import warnings
import json
from tarfile import TarFile
from pkgutil import get_data
from io import BytesIO
from dateutil.tz import tzfile as _tzfile
def gettz_db_metadata():
    """ Get the zonefile metadata

    See `zonefile_metadata`_

    :returns:
        A dictionary with the database metadata

    .. deprecated:: 2.6
        See deprecation warning in :func:`zoneinfo.gettz`. To get metadata,
        query the attribute ``zoneinfo.ZoneInfoFile.metadata``.
    """
    warnings.warn("zoneinfo.gettz_db_metadata() will be removed in future versions, to use the dateutil-provided zoneinfo files, ZoneInfoFile object and query the 'metadata' attribute instead. See the documentation for details.", DeprecationWarning)
    if len(_CLASS_ZONE_INSTANCE) == 0:
        _CLASS_ZONE_INSTANCE.append(ZoneInfoFile(getzoneinfofile_stream()))
    return _CLASS_ZONE_INSTANCE[0].metadata