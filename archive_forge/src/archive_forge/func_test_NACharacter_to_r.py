import pytest
import math
import rpy2.rinterface as ri
def test_NACharacter_to_r():
    na_character = ri.NA_Character
    assert ri.baseenv['is.na'](ri.StrSexpVector((na_character,)))[0]