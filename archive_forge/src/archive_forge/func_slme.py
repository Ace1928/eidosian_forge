from __future__ import annotations
import os
from math import pi
import matplotlib.pyplot as plt
import numpy as np
from scipy import constants
from scipy.constants import physical_constants, speed_of_light
from scipy.integrate import simps
from scipy.interpolate import interp1d
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.util.due import Doi, due
def slme(material_energy_for_absorbance_data, material_absorbance_data, material_direct_allowed_gap, material_indirect_gap, thickness=5e-05, temperature=293.15, absorbance_in_inverse_centimeters=False, cut_off_absorbance_below_direct_allowed_gap=True, plot_current_voltage=False):
    """
    Calculate the SLME.

    Args:
        material_energy_for_absorbance_data: energy grid for absorbance data
        material_absorbance_data: absorption coefficient in m^-1
        material_direct_allowed_gap: direct bandgap in eV
        material_indirect_gap: indirect bandgap in eV
        thickness: thickness of the material in m
        temperature: working temperature in K
        absorbance_in_inverse_centimeters: whether the absorbance data is in the unit of cm^-1
        cut_off_absorbance_below_direct_allowed_gap: whether to discard all absorption below bandgap
        plot_current_voltage: whether to plot the current-voltage curve

    Returns:
        The calculated maximum efficiency.
    """
    c = constants.c
    h = constants.h
    h_e = constants.h / constants.e
    k = constants.k
    k_e = constants.k / constants.e
    e = constants.e
    if absorbance_in_inverse_centimeters:
        material_absorbance_data = material_absorbance_data * 100
    solar_spectrum_data_file = str(os.path.join(os.path.dirname(__file__), 'am1.5G.dat'))
    solar_spectra_wavelength, solar_spectra_irradiance = np.loadtxt(solar_spectrum_data_file, usecols=[0, 1], unpack=True, skiprows=2)
    solar_spectra_wavelength_meters = solar_spectra_wavelength * 1e-09
    delta = material_direct_allowed_gap - material_indirect_gap
    fr = np.exp(-delta / (k_e * temperature))
    solar_spectra_photon_flux = solar_spectra_irradiance * (solar_spectra_wavelength_meters / (h * c))
    power_in = simps(solar_spectra_irradiance, solar_spectra_wavelength)
    blackbody_irradiance = 2.0 * h * c ** 2 / solar_spectra_wavelength_meters ** 5 * (1.0 / (np.exp(h * c / (solar_spectra_wavelength_meters * k * temperature)) - 1.0))
    blackbody_photon_flux = blackbody_irradiance * (solar_spectra_wavelength_meters / (h * c))
    material_wavelength_for_absorbance_data = c * h_e / (material_energy_for_absorbance_data + 1e-08) * 10 ** 9
    material_absorbance_data_function = interp1d(material_wavelength_for_absorbance_data, material_absorbance_data, kind='cubic', fill_value=(material_absorbance_data[0], material_absorbance_data[-1]), bounds_error=False)
    material_interpolated_absorbance = np.zeros(len(solar_spectra_wavelength_meters))
    for i in range(len(solar_spectra_wavelength_meters)):
        if solar_spectra_wavelength[i] < 1000000000.0 * (c * h_e / material_direct_allowed_gap) or cut_off_absorbance_below_direct_allowed_gap is False:
            material_interpolated_absorbance[i] = material_absorbance_data_function(solar_spectra_wavelength[i])
    absorbed_by_wavelength = 1.0 - np.exp(-2.0 * material_interpolated_absorbance * thickness)
    J_0_r = e * np.pi * simps(blackbody_photon_flux * absorbed_by_wavelength, solar_spectra_wavelength_meters)
    J_0 = J_0_r / fr
    J_sc = e * simps(solar_spectra_photon_flux * absorbed_by_wavelength, solar_spectra_wavelength)

    def J(V):
        return J_sc - J_0 * (np.exp(e * V / (k * temperature)) - 1.0)

    def power(V):
        return J(V) * V
    test_voltage = 0
    voltage_step = 0.001
    while power(test_voltage + voltage_step) > power(test_voltage):
        test_voltage += voltage_step
    max_power = power(test_voltage)
    efficiency = max_power / power_in
    if plot_current_voltage:
        V = np.linspace(0, 2, 200)
        plt.plot(V, J(V))
        plt.plot(V, power(V), linestyle='--')
        plt.savefig('pp.png')
        plt.close()
    return 100.0 * efficiency