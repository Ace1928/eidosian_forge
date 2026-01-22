import warnings
from ..water_viscosity_korson_1969 import water_viscosity
def test_water_viscosity():
    warnings.filterwarnings('error')
    assert abs(water_viscosity(273.15 + 0) - 1.7916) < 0.0005
    assert abs(water_viscosity(273.15 + 5) - 1.5192) < 0.0005
    assert abs(water_viscosity(273.15 + 10) - 1.3069) < 0.0005
    assert abs(water_viscosity(273.15 + 15) - 1.1382) < 0.0005
    assert abs(water_viscosity(273.15 + 20) - 1.002) < 0.0005
    assert abs(water_viscosity(273.15 + 25) - 0.8903) < 0.0005
    assert abs(water_viscosity(273.15 + 30) - 0.7975) < 0.0005
    assert abs(water_viscosity(273.15 + 35) - 0.7195) < 0.0005
    assert abs(water_viscosity(273.15 + 40) - 0.6532) < 0.0005
    assert abs(water_viscosity(273.15 + 45) - 0.5963) < 0.0005
    assert abs(water_viscosity(273.15 + 50) - 0.5471) < 0.0005
    assert abs(water_viscosity(273.15 + 55) - 0.5042) < 0.0005
    assert abs(water_viscosity(273.15 + 60) - 0.4666) < 0.0005
    assert abs(water_viscosity(273.15 + 65) - 0.4334) < 0.0005
    assert abs(water_viscosity(273.15 + 70) - 0.4039) < 0.0005
    assert abs(water_viscosity(273.15 + 75) - 0.3775) < 0.0005
    assert abs(water_viscosity(273.15 + 80) - 0.3538) < 0.0005
    assert abs(water_viscosity(273.15 + 85) - 0.3323) < 0.0005
    assert abs(water_viscosity(273.15 + 90) - 0.3128) < 0.0005
    assert abs(water_viscosity(273.15 + 95) - 0.2949) < 0.0006
    assert abs(water_viscosity(273.15 + 100) - 0.2783) < 0.002
    warnings.resetwarnings()