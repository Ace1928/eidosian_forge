from types import SimpleNamespace
from fontTools.misc.sstruct import calcsize, unpack, unpack2
Module for reading TFM (TeX Font Metrics) files.

The TFM format is described in the TFtoPL WEB source code, whose typeset form
can be found on `CTAN <http://mirrors.ctan.org/info/knuth-pdf/texware/tftopl.pdf>`_.

	>>> from fontTools.tfmLib import TFM
	>>> tfm = TFM("Tests/tfmLib/data/cmr10.tfm")
	>>>
	>>> # Accessing an attribute gets you metadata.
	>>> tfm.checksum
	1274110073
	>>> tfm.designsize
	10.0
	>>> tfm.codingscheme
	'TeX text'
	>>> tfm.family
	'CMR'
	>>> tfm.seven_bit_safe_flag
	False
	>>> tfm.face
	234
	>>> tfm.extraheader
	{}
	>>> tfm.fontdimens
	{'SLANT': 0.0, 'SPACE': 0.33333396911621094, 'STRETCH': 0.16666698455810547, 'SHRINK': 0.11111164093017578, 'XHEIGHT': 0.4305553436279297, 'QUAD': 1.0000028610229492, 'EXTRASPACE': 0.11111164093017578}
	>>> # Accessing a character gets you its metrics.
	>>> # “width” is always available, other metrics are available only when
	>>> # applicable. All values are relative to “designsize”.
	>>> tfm.chars[ord("g")]
	{'width': 0.5000019073486328, 'height': 0.4305553436279297, 'depth': 0.1944446563720703, 'italic': 0.013888359069824219}
	>>> # Kerning and ligature can be accessed as well.
	>>> tfm.kerning[ord("c")]
	{104: -0.02777862548828125, 107: -0.02777862548828125}
	>>> tfm.ligatures[ord("f")]
	{105: ('LIG', 12), 102: ('LIG', 11), 108: ('LIG', 13)}
