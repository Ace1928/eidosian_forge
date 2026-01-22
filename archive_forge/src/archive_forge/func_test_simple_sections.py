import typing as T
import pytest
from docstring_parser.numpydoc import compose, parse
def test_simple_sections() -> None:
    """Test parsing simple sections."""
    docstring = parse('\n        Short description\n\n        See Also\n        --------\n        something : some thing you can also see\n        actually, anything can go in this section\n\n        Warnings\n        --------\n        Here be dragons\n\n        Notes\n        -----\n        None of this is real\n\n        References\n        ----------\n        Cite the relevant literature, e.g. [1]_.  You may also cite these\n        references in the notes section above.\n\n        .. [1] O. McNoleg, "The integration of GIS, remote sensing,\n           expert systems and adaptive co-kriging for environmental habitat\n           modelling of the Highland Haggis using object-oriented, fuzzy-logic\n           and neural-network techniques," Computers & Geosciences, vol. 22,\n           pp. 585-588, 1996.\n        ')
    assert len(docstring.meta) == 4
    assert docstring.meta[0].args == ['see_also']
    assert docstring.meta[0].description == 'something : some thing you can also see\nactually, anything can go in this section'
    assert docstring.meta[1].args == ['warnings']
    assert docstring.meta[1].description == 'Here be dragons'
    assert docstring.meta[2].args == ['notes']
    assert docstring.meta[2].description == 'None of this is real'
    assert docstring.meta[3].args == ['references']