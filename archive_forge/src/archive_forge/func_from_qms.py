from __future__ import annotations
import json
import urllib.request
import uuid
from typing import Callable
from urllib.parse import quote
@classmethod
def from_qms(cls, name: str) -> TileProvider:
    """
        Creates a :class:`TileProvider` object based on the definition from
        the `Quick Map Services <https://qms.nextgis.com/>`__ open catalog.

        Parameters
        ----------
        name : str
            Service name

        Returns
        -------
        :class:`TileProvider`

        Examples
        --------
        >>> from xyzservices.lib import TileProvider
        >>> provider = TileProvider.from_qms("OpenTopoMap")
        """
    qms_api_url = 'https://qms.nextgis.com/api/v1/geoservices'
    services = json.load(urllib.request.urlopen(f'{qms_api_url}/?search={quote(name)}&type=tms'))
    for service in services:
        if service['name'] == name:
            break
    else:
        raise ValueError(f"Service '{name}' not found.")
    service_id = service['id']
    service_details = json.load(urllib.request.urlopen(f'{qms_api_url}/{service_id}'))
    return cls(name=service_details['name'], url=service_details['url'], min_zoom=service_details.get('z_min'), max_zoom=service_details.get('z_max'), attribution=service_details.get('copyright_text'))