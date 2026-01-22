import re
import json
import hashlib
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, ZoneAlreadyExistsError
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import InvalidCredsError, MalformedResponseError
from libcloud.common.exceptions import BaseHTTPError

        Update an existing record.

        :param record: Record object to update.
        :type  record: :class:`Record`

        :param name: name of the new record, for example "www".
        :type  name: ``str``

        :param type: DNS resource record type (A, AAAA, ...).
        :type  type: :class:`RecordType`

        :param data: Data for the record (depending on the record type).
        :type  data: ``str``

        :param extra: Extra attributes: 'ttl','disabled' (optional)
        :type   extra: ``dict``

        :rtype: :class:`Record`
        