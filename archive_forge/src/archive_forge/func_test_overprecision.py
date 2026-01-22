import matplotlib._type1font as t1f
import os.path
import difflib
import pytest
def test_overprecision():
    filename = os.path.join(os.path.dirname(__file__), 'cmr10.pfb')
    font = t1f.Type1Font(filename)
    slanted = font.transform({'slant': 0.167})
    lines = slanted.parts[0].decode('ascii').splitlines()
    matrix, = [line[line.index('[') + 1:line.index(']')] for line in lines if '/FontMatrix' in line]
    angle, = [word for line in lines if '/ItalicAngle' in line for word in line.split() if word[0] in '-0123456789']
    assert matrix == '0.001 0 0.000167 0.001 0 0'
    assert angle == '-9.4809'