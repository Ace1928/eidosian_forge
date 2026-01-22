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
def set_intent(self, code, params=(), name='', allow_unknown=False):
    """Set the intent code, parameters and name

        If parameters are not specified, assumed to be all zero. Each
        intent code has a set number of parameters associated. If you
        specify any parameters, then it will need to be the correct number
        (e.g the "f test" intent requires 2).  However, parameters can
        also be set in the file data, so we also allow not setting any
        parameters (empty parameter tuple).

        Parameters
        ----------
        code : integer or string
            code specifying nifti intent
        params : list, tuple of scalars
            parameters relating to intent (see intent_codes)
            defaults to ().  Unspecified parameters are set to 0.0
        name : string
            intent name (description). Defaults to ''
        allow_unknown : {False, True}, optional
            Allow unknown integer intent codes. If False (the default),
            a KeyError is raised on attempts to set the intent
            to an unknown code.

        Returns
        -------
        None

        Examples
        --------
        >>> hdr = Nifti1Header()
        >>> hdr.set_intent(0)  # no intent
        >>> hdr.set_intent('z score')
        >>> hdr.get_intent()
        ('z score', (), '')
        >>> hdr.get_intent('code')
        (5, (), '')
        >>> hdr.set_intent('t test', (10,), name='some score')
        >>> hdr.get_intent()
        ('t test', (10.0,), 'some score')
        >>> hdr.set_intent('f test', (2, 10), name='another score')
        >>> hdr.get_intent()
        ('f test', (2.0, 10.0), 'another score')
        >>> hdr.set_intent('f test')
        >>> hdr.get_intent()
        ('f test', (0.0, 0.0), '')
        >>> hdr.set_intent(9999, allow_unknown=True) # unknown code
        >>> hdr.get_intent()
        ('unknown code 9999', (), '')
        """
    hdr = self._structarr
    known_intent = code in intent_codes
    if not known_intent:
        if not allow_unknown or isinstance(code, str):
            raise KeyError('Unknown intent code: ' + str(code))
    if known_intent:
        icode = intent_codes.code[code]
        p_descr = intent_codes.parameters[code]
    else:
        icode = code
        p_descr = ('p1', 'p2', 'p3')
    if len(params) and len(params) != len(p_descr):
        raise HeaderDataError(f'Need params of form {p_descr}, or empty')
    hdr['intent_code'] = icode
    hdr['intent_name'] = name
    all_params = [0] * 3
    all_params[:len(params)] = params[:]
    for i, param in enumerate(all_params):
        hdr['intent_p%d' % (i + 1)] = param