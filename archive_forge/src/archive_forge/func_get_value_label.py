from __future__ import annotations
import numpy as np
from . import imageglobals as imageglobals
from .batteryrunners import BatteryRunner
from .volumeutils import Recoder, endian_codes, native_code, pretty_mapping, swapped_code
def get_value_label(self, fieldname):
    """Returns label for coded field

        A coded field is an int field containing codes that stand for
        discrete values that also have string labels.

        Parameters
        ----------
        fieldname : str
           name of header field to get label for

        Returns
        -------
        label : str
           label for code value in header field `fieldname`

        Raises
        ------
        ValueError
            if field is not coded.

        Examples
        --------
        >>> from nibabel.volumeutils import Recoder
        >>> recoder = Recoder(((1, 'one'), (2, 'two')), ('code', 'label'))
        >>> class C(LabeledWrapStruct):
        ...     template_dtype = np.dtype([('datatype', 'i2')])
        ...     _field_recoders = dict(datatype = recoder)
        >>> hdr  = C()
        >>> hdr.get_value_label('datatype')
        '<unknown code 0>'
        >>> hdr['datatype'] = 2
        >>> hdr.get_value_label('datatype')
        'two'
        """
    if fieldname not in self._field_recoders:
        raise ValueError(f'{fieldname} not a coded field')
    code = int(self._structarr[fieldname])
    try:
        return self._field_recoders[fieldname].label[code]
    except KeyError:
        return f'<unknown code {code}>'