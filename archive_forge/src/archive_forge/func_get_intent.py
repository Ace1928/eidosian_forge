from __future__ import annotations
import warnings
from io import BytesIO
import numpy as np
import numpy.linalg as npl
from . import analyze  # module import
from .arrayproxy import get_obj_dtype
from .batteryrunners import Report
from .casting import have_binary128
from .deprecated import alert_future_error
from .filebasedimages import ImageFileError, SerializableImage
from .optpkg import optional_package
from .quaternions import fillpositive, mat2quat, quat2mat
from .spatialimages import HeaderDataError
from .spm99analyze import SpmAnalyzeHeader
from .volumeutils import Recoder, endian_codes, make_dt_codes
def get_intent(self, code_repr='label'):
    """Get intent code, parameters and name

        Parameters
        ----------
        code_repr : string
           string giving output form of intent code representation.
           Default is 'label'; use 'code' for integer representation.

        Returns
        -------
        code : string or integer
            intent code, or string describing code
        parameters : tuple
            parameters for the intent
        name : string
            intent name

        Examples
        --------
        >>> hdr = Nifti1Header()
        >>> hdr.set_intent('t test', (10,), name='some score')
        >>> hdr.get_intent()
        ('t test', (10.0,), 'some score')
        >>> hdr.get_intent('code')
        (3, (10.0,), 'some score')
        """
    hdr = self._structarr
    recoder = self._field_recoders['intent_code']
    code = int(hdr['intent_code'])
    known_intent = code in recoder
    if code_repr == 'code':
        label = code
    elif code_repr == 'label':
        if known_intent:
            label = recoder.label[code]
        else:
            label = 'unknown code ' + str(code)
    else:
        raise TypeError('repr can be "label" or "code"')
    n_params = len(recoder.parameters[code]) if known_intent else 0
    params = (float(hdr['intent_p%d' % (i + 1)]) for i in range(n_params))
    name = hdr['intent_name'].item().decode('latin-1')
    return (label, tuple(params), name)