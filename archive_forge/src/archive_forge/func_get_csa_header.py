import numpy as np
from .structreader import Unpacker
from .utils import find_private_section
def get_csa_header(dcm_data, csa_type='image'):
    """Get CSA header information from DICOM header

    Return None if the header does not contain CSA information of the
    specified `csa_type`

    Parameters
    ----------
    dcm_data : dicom.Dataset
       DICOM dataset.  Should implement ``__getitem__`` and, if initial check
       for presence of ``dcm_data[(0x29, 0x10)]`` passes, should satisfy
       interface for ``find_private_section``.
    csa_type : {'image', 'series'}, optional
       Type of CSA field to read; default is 'image'

    Returns
    -------
    csa_info : None or dict
       Parsed CSA field of `csa_type` or None, if we cannot find the CSA
       information.
    """
    csa_type = csa_type.lower()
    if csa_type == 'image':
        element_offset = 16
    elif csa_type == 'series':
        element_offset = 32
    else:
        raise ValueError(f'Invalid CSA header type "{csa_type}"')
    if not (41, 16) in dcm_data:
        return None
    section_start = find_private_section(dcm_data, 41, 'SIEMENS CSA HEADER')
    if section_start is None:
        return None
    element_no = section_start + element_offset
    try:
        tag = dcm_data[41, element_no]
    except KeyError:
        return None
    return read(tag.value)