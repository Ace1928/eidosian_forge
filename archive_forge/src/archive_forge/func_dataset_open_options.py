import logging
import xml.etree.ElementTree as ET
from fiona.env import require_gdal_version
from fiona.ogrext import _get_metadata_item
@require_gdal_version('2.0')
def dataset_open_options(driver):
    """ Returns dataset open options for driver

    Parameters
    ----------
    driver : str

    Returns
    -------
    dict
        Dataset open options

    """
    xml = _get_metadata_item(driver, MetadataItem.DATASET_OPEN_OPTIONS)
    if xml is None:
        return {}
    if len(xml) == 0:
        return {}
    return _parse_options(xml)