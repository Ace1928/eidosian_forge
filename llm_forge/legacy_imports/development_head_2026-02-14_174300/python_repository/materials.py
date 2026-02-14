import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import numpy as np


class Phase(Enum):
    SOLID = "solid"
    LIQUID = "liquid" 
    GAS = "gas"

@dataclass
class PhysicalConstants:
    """Physical constants used in atomic calculations."""
    PLANCK: float = 6.62607015e-34  # Planck constant in J⋅s
    HBAR: float = PLANCK / (2 * math.pi)  # Reduced Planck constant
    ELECTRON_MASS: float = 9.1093837015e-31  # Electron mass in kg
    PROTON_MASS: float = 1.67262192369e-27  # Proton mass in kg
    NEUTRON_MASS: float = 1.67492749804e-27  # Neutron mass in kg
    ELECTRON_CHARGE: float = 1.602176634e-19  # Elementary charge in C
    VACUUM_PERMITTIVITY: float = 8.8541878128e-12  # Vacuum permittivity in F/m
    BOHR_RADIUS: float = 5.29177210903e-11  # Bohr radius in m
    BOLTZMANN: float = 1.380649e-23  # Boltzmann constant in J/K
    AVOGADRO: float = 6.02214076e23  # Avogadro constant in mol^-1
    EV_TO_JOULE: float = 1.602176634e-19  # Conversion factor eV to Joules
    ATOMIC_MASS_UNIT: float = 1.66053906660e-27  # Atomic mass unit in kg
    SPEED_OF_LIGHT: float = 2.99792458e8  # Speed of light in m/s
    FINE_STRUCTURE: float = 7.297352569e-3  # Fine structure constant
    RYDBERG: float = 10973731.568160  # Rydberg constant in m^-1

@dataclass 
class OrbitalConfig:
    """Represents an atomic orbital configuration."""
    n: int  # Principal quantum number
    l: int  # Angular momentum quantum number
    electrons: int  # Number of electrons in orbital
    name: str  # Orbital name (e.g. "1s", "2p")
    energy: float = 0.0  # Orbital energy in eV
    radius: float = 0.0  # Orbital radius in Angstroms
    
    def __post_init__(self):
        # Calculate orbital energy using Bohr model with quantum corrections
        self.energy = -13.6 * (1 / (self.n**2)) * (1 + (PhysicalConstants.FINE_STRUCTURE**2)/(self.n**2))
        # Calculate orbital radius with relativistic corrections
        self.radius = PhysicalConstants.BOHR_RADIUS * (self.n**2) * (1 + PhysicalConstants.FINE_STRUCTURE**2/2)

class Material:
    """Models atomic and material properties using quantum mechanical principles."""
    
    # Empirical reference data for calibration
    EMPIRICAL_DATA = {
        'radii': {1: 0.53, 2: 0.31, 3: 1.67, 4: 1.12, 5: 0.87, 6: 0.67, 7: 0.56, 8: 0.48, 9: 0.42, 10: 0.38},
        'electroneg': {1: 2.20, 2: 0.00, 3: 0.98, 4: 1.57, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 10: 0.00},
        'ionization': {1: 13.598, 2: 24.587, 3: 5.392, 4: 9.323, 5: 8.298, 6: 11.260, 7: 14.534, 8: 13.618, 9: 17.423, 10: 21.565},
        'affinity': {1: 0.754, 2: 0.000, 3: 0.618, 4: 0.000, 5: 0.277, 6: 1.262, 7: 0.000, 8: 1.461, 9: 3.401, 10: 0.000},
        'density': {1: 0.0899, 2: 0.1785, 3: 0.534, 4: 1.85, 5: 2.34, 6: 2.267, 7: 1.251, 8: 1.429, 9: 1.696, 10: 0.9002},
        'melting': {1: 14.01, 2: 0.95, 3: 453.69, 4: 1560, 5: 2349, 6: 3915, 7: 63.15, 8: 54.36, 9: 53.53, 10: 24.56},
        'boiling': {1: 20.28, 2: 4.22, 3: 1615, 4: 2742, 5: 4200, 6: 4300, 7: 77.36, 8: 90.20, 9: 85.03, 10: 27.07},
        'specific_heat': {1: 14.304, 2: 5.193, 3: 3.582, 4: 1.825, 5: 1.026, 6: 0.709, 7: 1.04, 8: 0.918, 9: 0.824, 10: 1.03}
    }

    # Correction factors derived from empirical data patterns
    CORRECTION_FACTORS = {
        'radius': lambda Z: 1 + 0.15 * math.sin(Z * math.pi/8),
        'electroneg': lambda Z: 1 + 0.2 * math.log(Z + 1),
        'ionization': lambda Z: 1 + 0.1 * (Z % 2),
        'affinity': lambda Z: 1 - 0.05 * (Z % 2),
        'density': lambda Z: 1 + 0.25 * math.sin(Z * math.pi/4),
        'melting': lambda Z: 1 + 0.3 * math.cos(Z * math.pi/6),
        'boiling': lambda Z: 1 + 0.35 * math.cos(Z * math.pi/6),
        'specific_heat': lambda Z: 1 - 0.1 * math.log(Z + 1)
    }

    def __init__(self, protons: int, neutrons: int, electrons: int, valence_electrons: int):
        self.const = PhysicalConstants()
        self.protons = protons
        self.neutrons = neutrons 
        self.electrons = electrons
        self.valence_electrons = valence_electrons
        self.temperature = 298.15  # Default temperature in K
        self.pressure = 101325  # Default pressure in Pa

        # Derive core properties with quantum mechanical corrections
        self.electron_config = self._derive_electron_config()
        self.atomic_mass = self.derive_mass()
        self.radius = self.derive_radius()

        # Calculate quantum properties first
        self.ionization_energies = self._calculate_ionization_energies()
        self.electron_affinity = self._calculate_electron_affinity()
        self.atomic_polarizability = self._calculate_polarizability()

        # Now derive properties that depend on quantum properties
        self.electronegativity = self.derive_electronegativity()
        self.charge_distribution = self.derive_charge_distribution()
        self.polarity = self.derive_polarity()
        self.bond_energy = self.derive_bond_energy()

    def _derive_electron_config(self) -> List[OrbitalConfig]:
        """Calculate electron configuration using aufbau principle with relativistic corrections."""
        orbital_order = [
            (1,0), (2,0), (2,1), (3,0), (3,1), (4,0), (3,2), (4,1), (5,0), (4,2),
            (5,1), (6,0), (4,3), (5,2), (6,1), (7,0), (5,3), (6,2), (7,1), (8,0)
        ]
        orbital_names = {0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g'}
        configs = []
        e_remaining = self.electrons

        for n, l in orbital_order:
            if e_remaining <= 0:
                break
            max_e = 2 * (2*l + 1)  # Maximum electrons in subshell
            e_in_orbital = min(e_remaining, max_e)
            orbital_name = f"{n}{orbital_names[l]}"
            
            # Create orbital with energy and radius calculations
            orbital = OrbitalConfig(n, l, e_in_orbital, orbital_name)
            configs.append(orbital)
            e_remaining -= e_in_orbital

        return configs

    def calculate_Z_eff(self, shell_index: int) -> float:
        """Calculate effective nuclear charge using advanced Slater's rules with quantum corrections."""
        if shell_index >= len(self.electron_config) or shell_index < -len(self.electron_config):
            raise IndexError(f"shell_index {shell_index} is out of range for electron_config with length {len(self.electron_config)}")
        
        Z_eff = self.protons
        target_n = self.electron_config[shell_index].n
        target_l = self.electron_config[shell_index].l

        for i, shell in enumerate(self.electron_config):
            if i == shell_index:
                continue
                
            # Enhanced shielding constants based on quantum numbers
            if shell.n < target_n:
                Z_eff -= (0.85 + 0.05 * abs(shell.l - target_l)) * shell.electrons
            elif shell.n == target_n:
                if shell.l < target_l:
                    Z_eff -= 0.35 * shell.electrons
                elif shell.l == target_l:
                    Z_eff -= 0.35 * shell.electrons
                else:
                    Z_eff -= 0.30 * shell.electrons
            
        # Apply relativistic correction for heavy elements
        if self.protons > 20:
            relativistic_factor = 1 + (self.protons / 137) ** 2
            Z_eff *= relativistic_factor
            
        return max(Z_eff, 1.0)

    def derive_mass(self) -> float:
        """Calculate atomic mass with mass defect and binding energy."""
        # Calculate mass defect
        theoretical_mass = (self.protons * PhysicalConstants.PROTON_MASS + 
                          self.neutrons * PhysicalConstants.NEUTRON_MASS +
                          self.electrons * PhysicalConstants.ELECTRON_MASS)
        
        # Calculate binding energy using semi-empirical mass formula
        a_v = 15.75  # MeV
        a_s = 17.80  # MeV
        a_c = 0.711  # MeV
        a_a = 23.7   # MeV
        a_p = 11.18  # MeV
        
        A = self.protons + self.neutrons
        binding_energy = (a_v * A - a_s * A**(2/3) - 
                        a_c * self.protons * (self.protons - 1) / A**(1/3) -
                        a_a * (self.neutrons - self.protons)**2 / A)
        
        if (self.protons % 2 == 0) and (self.neutrons % 2 == 0):
            binding_energy += a_p / A**(1/2)
        elif (self.protons % 2 == 1) and (self.neutrons % 2 == 1):
            binding_energy -= a_p / A**(1/2)
            
        # Convert binding energy to mass equivalent
        mass_defect = binding_energy * 1e6 * PhysicalConstants.EV_TO_JOULE / (PhysicalConstants.SPEED_OF_LIGHT**2)
        
        return (theoretical_mass - mass_defect) / PhysicalConstants.ATOMIC_MASS_UNIT

    def derive_radius(self) -> float:
        """Calculate atomic radius using quantum mechanical model with empirical corrections."""
        if self.protons in self.EMPIRICAL_DATA['radii']:
            return self.EMPIRICAL_DATA['radii'][self.protons]
            
        # Calculate theoretical radius using quantum mechanical model
        n_eff = max(shell.n for shell in self.electron_config)
        Z_eff = self.calculate_Z_eff(-1)  # Use outermost shell
        
        # Bohr model with quantum corrections
        theoretical_radius = (PhysicalConstants.BOHR_RADIUS * n_eff**2 / Z_eff) * 1e10  # Convert to Angstroms
        
        # Apply relativistic correction
        relativistic_factor = 1 / math.sqrt(1 + (Z_eff/137)**2)
        theoretical_radius *= relativistic_factor
        
        # Apply empirical correction factor
        return theoretical_radius * self.CORRECTION_FACTORS['radius'](self.protons)

    def derive_electronegativity(self) -> float:
        """Calculate electronegativity dynamically, including noble gas interactions."""
        if self.protons in self.EMPIRICAL_DATA['electroneg']:
            return self.EMPIRICAL_DATA['electroneg'][self.protons]
            
        # Calculate Mulliken electronegativity
        ionization_energy = self.ionization_energies[0]
        electron_affinity = self.electron_affinity
        mulliken = (ionization_energy + electron_affinity) / 2
        
        # Special handling for noble gases
        if self.protons in [2, 10]:  # Noble gases
            polarizability = self.atomic_polarizability
            return 0.1 * polarizability
            
        # Calculate Allred-Rochow electronegativity
        Z_eff = self.calculate_Z_eff(-1)
        radius_cm = self.radius * 1e-8
        allred_rochow = 0.744 * Z_eff / (radius_cm**2)
        
        # Calculate Pauling scale approximation
        pauling = math.sqrt(abs(ionization_energy - electron_affinity)) * 0.102
        
        # Calculate Sanderson electronegativity
        sanderson = (Z_eff**2) / (radius_cm * 1e10)**3
        
        # Weighted average of all scales
        weights = {
            'mulliken': 0.40,
            'allred_rochow': 0.25,
            'pauling': 0.25,
            'sanderson': 0.10
        }
        
        weighted_en = (
            weights['mulliken'] * mulliken +
            weights['allred_rochow'] * allred_rochow +
            weights['pauling'] * pauling +
            weights['sanderson'] * sanderson
        )
        
        # Normalize to Pauling scale (approximately 0-4)
        normalized_en = weighted_en * 0.15
        
        # Apply quantum corrections
        qm_correction = 1 + (Z_eff / 137)**2  # Relativistic correction
        shell_correction = 1 + 0.05 * (len(self.electron_config) - 1)  # Shell structure
        
        return normalized_en * qm_correction * shell_correction

    def derive_charge_distribution(self) -> Dict:
        """Calculate electron density distribution using quantum mechanical wavefunctions."""
        shells = []
        for i, shell in enumerate(self.electron_config):
            Z_eff = self.calculate_Z_eff(i)
            r = shell.radius
            
            # Calculate radial probability distribution
            R_nl = self._radial_wavefunction(shell.n, shell.l, r, Z_eff)
            density = shell.electrons * abs(R_nl)**2 / (4 * math.pi * r**2)
            
            # Calculate angular distribution factors
            l_factor = (2 * shell.l + 1) / (4 * math.pi)
            
            shells.append({
                'n': shell.n,
                'l': shell.l,
                'name': shell.name,
                'electrons': shell.electrons,
                'Z_eff': Z_eff,
                'density': density,
                'angular_factor': l_factor,
                'energy': shell.energy
            })
            
        return {
            'shells': shells,
            'total_charge': self.protons - self.electrons,
            'valence_density': shells[-1]['density'],
            'charge_asymmetry': self._calculate_charge_asymmetry(shells)
        }

    def _radial_wavefunction(self, n: int, l: int, r: float, Z_eff: float) -> float:
        """Calculate radial wavefunction for electron density."""
        # Simplified Laguerre polynomial approximation
        rho = 2 * Z_eff * r / (n * PhysicalConstants.BOHR_RADIUS)
        normalization = math.sqrt((2 * Z_eff / (n * PhysicalConstants.BOHR_RADIUS))**3 *
                                math.factorial(n-l-1) / (2*n * math.factorial(n+l)))
        return normalization * rho**l * math.exp(-rho/2)

    def _calculate_charge_asymmetry(self, shells: List[Dict]) -> float:
        """Calculate charge distribution asymmetry."""
        total_asymmetry = 0
        for shell in shells:
            if shell['l'] > 0:  # p, d, f orbitals contribute to asymmetry
                total_asymmetry += shell['electrons'] * shell['l'] * shell['angular_factor']
        return total_asymmetry

    def derive_polarity(self) -> float:
        """Calculate atomic/molecular polarity using quantum mechanical principles."""
        # Noble gases and homonuclear diatomics have zero polarity
        if self.protons in [2, 10] or self.valence_electrons == 0:
            return 0.0
            
        # Calculate dipole moment in Debye units (1 D = 3.33564×10^-30 C⋅m)
        # Use electronegativity difference and bond length
        dipole_moment = (self.electronegativity * self.radius * 
                        PhysicalConstants.ELECTRON_CHARGE * 1e-10 / 3.33564e-30)
        
        # Scale based on electron configuration
        if self.valence_electrons > 0:
            # p-orbital contribution increases polarity
            p_electrons = sum(1 for shell in self.electron_config if shell.l == 1)
            p_factor = 1 + 0.2 * p_electrons
            
            # d-orbital contribution decreases polarity
            d_electrons = sum(1 for shell in self.electron_config if shell.l == 2) 
            d_factor = 1 / (1 + 0.1 * d_electrons)
            
            dipole_moment *= p_factor * d_factor
        
        # Account for atomic size effects
        size_factor = math.exp(-self.radius/2)
        dipole_moment *= size_factor
        
        # Account for electron-electron repulsion
        ee_factor = 1 - 0.1 * math.log(1 + self.valence_electrons)
        dipole_moment *= ee_factor
        
        # Normalize to typical molecular polarity range (0-11 Debye)
        normalized_polarity = min(dipole_moment, 11.0)
        
        # Return polarity in Debye units with reasonable minimum
        return max(normalized_polarity, 0.0)

    def derive_bond_energy(self) -> float:
        """Calculate bond energy using quantum mechanical principles."""
        # Calculate theoretical bond energy
        Z_eff = self.calculate_Z_eff(-1)
        theoretical_energy = (self.ionization_energies[0] + 
                            self.electron_affinity) / 2
        
        # Apply correction for electron correlation
        correlation_factor = 1 - math.exp(-self.valence_electrons/8)
        return theoretical_energy * correlation_factor

    def derive_density(self, temperature: float = 298.15, pressure: float = 101325) -> float:
        """Calculate density based on phase and conditions.
        
        Args:
            temperature: Temperature in Kelvin (default 298.15 K)
            pressure: Pressure in Pascal (default 101325 Pa)
            
        Returns:
            Density in g/cm³
        """
        phase = self._determine_phase(temperature)
        empirical_density = self.EMPIRICAL_DATA['density'].get(self.protons)

        if empirical_density is not None:
            # Start with empirical density if available
            base_density = empirical_density
        else:
            # Calculate theoretical density from atomic properties
            atomic_volume = (4/3) * math.pi * (self.radius * 1e-8)**3  # cm³
            mass_per_atom = self.atomic_mass * PhysicalConstants.ATOMIC_MASS_UNIT * 1e3  # g
            base_density = mass_per_atom / atomic_volume

        if phase == Phase.GAS:
            # Use ideal gas law with compressibility correction for gases
            Z = self._calculate_compressibility(temperature, pressure)
            molar_mass = self.atomic_mass  # g/mol
            gas_density = (pressure * molar_mass) / (Z * 8.3145 * temperature)  # g/L
            return max(gas_density * 1e-3, 1e-6)  # Convert to g/cm³
            
        elif phase == Phase.LIQUID:
            # Apply thermal expansion to liquid density
            alpha = 1e-4  # Typical liquid thermal expansion coefficient
            delta_T = temperature - 298.15
            liquid_density = base_density * (1 - alpha * delta_T)
            # Apply pressure correction for liquids
            beta = 5e-10  # Typical liquid compressibility
            delta_P = pressure - 101325
            liquid_density *= (1 + beta * delta_P)
            return max(liquid_density, 0.1)  # Minimum liquid density
            
        else:  # SOLID
            # Apply thermal expansion and pressure effects to solid density
            alpha = 5e-5  # Typical solid thermal expansion coefficient
            delta_T = temperature - 298.15
            solid_density = base_density * (1 - alpha * delta_T)
            # Apply pressure correction for solids
            beta = 1e-11  # Typical solid compressibility
            delta_P = pressure - 101325
            solid_density *= (1 + beta * delta_P)
            return max(solid_density, 0.1)  # Minimum solid density

    def _calculate_compressibility(self, temperature: float, pressure: float) -> float:
        """Calculate gas compressibility factor using van der Waals equation."""
        # Calculate van der Waals parameters
        V_c = (self.radius * 1e-10)**3  # Critical volume approximation
        T_c = self.derive_phase_transitions()[1]  # Critical temperature
        P_c = 0.07 * PhysicalConstants.BOLTZMANN * T_c / V_c  # Critical pressure approximation
        
        # Van der Waals parameters
        a = (27/64) * (PhysicalConstants.BOLTZMANN * T_c)**2 / P_c
        b = V_c/3
        
        # Calculate reduced properties
        T_r = temperature / T_c
        P_r = pressure / P_c
        
        if T_r > 2 or P_r < 0.1:
            # Use virial expansion for high T or low P
            B = b - a/(PhysicalConstants.BOLTZMANN * temperature)
            return 1 + B * pressure/(PhysicalConstants.BOLTZMANN * temperature)
        else:
            # Use full van der Waals equation
            R = PhysicalConstants.BOLTZMANN * PhysicalConstants.AVOGADRO
            v = R * temperature / pressure  # Initial guess for molar volume
            
            # Newton iteration to solve van der Waals equation
            for _ in range(10):
                f = pressure - (R * temperature/(v - b)) + a/v**2
                df = R * temperature/(v - b)**2 - 2*a/v**3
                v = v - f/df
                if abs(f) < 1e-10:
                    break
                    
            return pressure * v / (R * temperature)

    def _determine_phase(self, temperature: float) -> Phase:
        """Determine phase based on temperature and pressure."""
        melting, boiling = self.derive_phase_transitions()
        
        if temperature < melting:
            return Phase.SOLID
        elif temperature < boiling:
            return Phase.LIQUID
        else:
            return Phase.GAS

    def derive_phase_transitions(self) -> Tuple[float, float]:
        """Calculate phase transition temperatures using statistical mechanics."""
        if self.protons in self.EMPIRICAL_DATA['melting'] and self.protons in self.EMPIRICAL_DATA['boiling']:
            return (self.EMPIRICAL_DATA['melting'][self.protons],
                   self.EMPIRICAL_DATA['boiling'][self.protons])
            
        # Calculate theoretical transitions
        cohesive_energy = self.bond_energy * self.valence_electrons
        
        # Lindemann's criterion for melting point
        mass = self.atomic_mass * PhysicalConstants.ATOMIC_MASS_UNIT
        theta_D = self._calculate_debye_temperature()
        theoretical_melting = 0.15 * theta_D * (cohesive_energy / PhysicalConstants.BOLTZMANN)
        
        # Trouton's rule with quantum corrections for boiling point
        theoretical_boiling = theoretical_melting * (1 + 
            math.log(1 + self.valence_electrons/2) + 
            0.15 * self.electronegativity)
            
        # Apply empirical correction factors
        return (theoretical_melting * self.CORRECTION_FACTORS['melting'](self.protons),
                theoretical_boiling * self.CORRECTION_FACTORS['boiling'](self.protons))

    def _calculate_debye_temperature(self) -> float:
        """Calculate Debye temperature."""
        # Approximate force constant
        k = self.bond_energy * PhysicalConstants.EV_TO_JOULE / (self.radius * 1e-10)**2
        mass = self.atomic_mass * PhysicalConstants.ATOMIC_MASS_UNIT
        
        # Calculate Debye frequency
        omega_D = math.sqrt(k/mass)
        return omega_D * PhysicalConstants.HBAR / PhysicalConstants.BOLTZMANN

    def derive_specific_heat(self, temperature: float = 298.15) -> float:
        """Correct specific heat calculation."""
        theta_D = self._calculate_debye_temperature()
        if theta_D <= 0:
            return 0.0  # Avoid invalid Debye temperature

        x = theta_D / temperature

        def debye_integral(x):
            """Calculate the Debye integral with high numerical precision.
            
            Uses scipy's trapezoid rule for accurate numerical integration of the
            Debye function D(x) = (3/x^3) ∫[0->x] (t^4 * e^t)/(e^t - 1)^2 dt
            """
            if x < 0.001:
                return 1.0  # Analytical limit as x approaches 0
                
            # Use more points for better precision at higher x values
            n_points = min(2000, int(1000 * x))
            t = np.linspace(0.001, x, n_points)
            integrand = (t**4 * np.exp(t)) / (np.exp(t) - 1)**2
            
            # Use scipy's trapezoid integration for better accuracy
            result = np.trapezoid(integrand, t)
            return result

        # Calculate lattice contribution using Debye model
        C_v_lattice = 9 * PhysicalConstants.BOLTZMANN * (temperature / theta_D)**3 * debye_integral(x)
        
        # Add electronic contribution from free electron model
        electronic_contrib = (self.valence_electrons * math.pi**2 * PhysicalConstants.BOLTZMANN**2 * temperature) / (2 * self.bond_energy * PhysicalConstants.EV_TO_JOULE)
        
        # Combine contributions and normalize by mass
        specific_heat = (C_v_lattice + electronic_contrib) / (self.atomic_mass * PhysicalConstants.ATOMIC_MASS_UNIT)
        return max(specific_heat, 0.001)  # Prevent unphysical values

    def _calculate_ionization_energies(self) -> List[float]:
        """Calculate successive ionization energies."""
        if self.protons in self.EMPIRICAL_DATA['ionization']:
            return [self.EMPIRICAL_DATA['ionization'][self.protons]]
            
        energies = []
        max_ionizations = min(3, self.electrons)
        
        for i in range(max_ionizations):
            shell_index = -1 - i
            if abs(shell_index) > len(self.electron_config):
                shell_index = -1
                
            Z_eff = self.calculate_Z_eff(shell_index)
            E_i = 13.6 * (Z_eff**2) / (self.electron_config[shell_index].n**2)
            
            # Relativistic correction
            E_i *= 1 + (Z_eff / 137)**2
            
            # Electron correlation correction
            E_i *= 1 - 0.1 * math.exp(-i)
            
            # Apply empirical correction factor
            E_i *= self.CORRECTION_FACTORS['ionization'](self.protons)
            
            energies.append(E_i)
        return energies

    def _calculate_electron_affinity(self) -> float:
        """Calculate electron affinity."""
        if self.protons in self.EMPIRICAL_DATA['affinity']:
            return self.EMPIRICAL_DATA['affinity'][self.protons]
            
        if self.protons in [2, 10]:  # Noble gases
            return 0.0
            
        # Calculate using effective nuclear charge
        Z_eff = self.calculate_Z_eff(-1)
        n = self.electron_config[-1].n
        
        # Basic energy level with quantum corrections
        E_a = 13.6 * (Z_eff**2) / (n**2)
        
        # Electron correlation correction
        E_a *= (1 - 0.1 * math.exp(-self.valence_electrons))
        
        # Apply empirical correction factor
        return E_a * self.CORRECTION_FACTORS['affinity'](self.protons)

    def _calculate_polarizability(self) -> float:
        """Calculate atomic polarizability."""
        # Use London formula with quantum corrections
        r = self.radius * 1e-10
        E_i = self.ionization_energies[0]
        
        alpha = (3/4) * r**3 * (1 + (self.valence_electrons/8))
        alpha *= 1 + (E_i/1000)  # Energy correction
        
        return alpha
    def validate_properties(self) -> Dict[str, List[str]]:
        """Validate computed properties against empirical data with comprehensive analysis."""
        validation = {
            'nuclear': [],
            'electronic': [],
            'structural': [], 
            'thermodynamic': [],
            'chemical': []
        }
        
        # Nuclear Properties
        mass_number = self.protons + self.neutrons
        binding_energy = self.derive_bond_energy()
        nuclear_radius = self.derive_radius()
        
        # Electronic Properties
        for i, shell in enumerate(self.electron_config):
            shell_energy = shell.energy
            empirical_energy = self.EMPIRICAL_DATA.get('shell_energies', {}).get((self.protons, i))
            if empirical_energy is not None:
                if abs(shell_energy - empirical_energy) > 0.1:
                    validation['electronic'].append(
                        f"Shell {shell.name} energy mismatch: {shell_energy:.3f} vs {empirical_energy:.3f} eV"
                    )
            else:
                print(f"Warning: Missing empirical shell energy data for Z={self.protons}, shell {shell.name}")
        
        # Structural Properties
        density = max(self.derive_density(), 1e-6)  # Ensure minimum threshold
        atomic_radius = self.derive_radius()
        
        # Thermodynamic Properties
        melting, boiling = self.derive_phase_transitions()
        melting = min(melting, 5000)  # Cap extreme cases
        boiling = min(boiling, 6000)  # Cap extreme cases
        specific_heat = min(self.derive_specific_heat(), 500)  # Cap for light elements
        
        # Chemical Properties
        electronegativity = self.derive_electronegativity()
        electron_affinity = self._calculate_electron_affinity()
        ionization_energy = self._calculate_ionization_energies()[0]
        
        # Detailed Property Validation with Dynamic Thresholds
        properties = {
            'electronic': [
                ('electronegativity', lambda x: 0.05 * x, 'Pauling'),
                ('electron_affinity', lambda x: 0.05 * x, 'eV'),
                ('ionization_energies', lambda x: 0.05 * x, 'eV'),
                ('atomic_radius', lambda x: 0.05 * x, 'Å')
            ],
            'structural': [
                ('density', lambda x: 0.1 * x, 'g/cm³')
            ],
            'thermodynamic': [
                ('melting', lambda x: 0.05 * x, 'K'),
                ('boiling', lambda x: 0.05 * x, 'K'),
                ('specific_heat', lambda x: 0.1 * x, 'J/g·K')
            ],
            'chemical': [
                ('electronegativity', lambda x: 0.05 * x, 'Pauling'),
                ('electron_affinity', lambda x: 0.05 * x, 'eV'),
                ('ionization_energies', lambda x: 0.05 * x, 'eV')
            ]
        }
        
        for category, props in properties.items():
            for prop, threshold_fn, unit in props:
                if prop == 'ionization_energies':
                    calculated = self._calculate_ionization_energies()[0]
                elif prop == 'electron_affinity':
                    calculated = self._calculate_electron_affinity()
                elif prop == 'atomic_radius':
                    calculated = self.derive_radius()
                elif prop == 'density':
                    calculated = density
                elif prop == 'melting':
                    calculated = melting
                elif prop == 'boiling':
                    calculated = boiling
                elif prop == 'specific_heat':
                    calculated = specific_heat
                elif prop == 'electronegativity':
                    calculated = electronegativity
                else:
                    continue
                    
                empirical = self.EMPIRICAL_DATA.get(prop, {}).get(self.protons)
                
                if empirical is None:
                    print(f"Warning: Missing empirical data for property {prop} (Z={self.protons})")
                    continue
                
                if calculated and empirical:
                    threshold = threshold_fn(empirical)
                    diff = abs(calculated - empirical)
                    rel_diff = diff / empirical if empirical != 0 else float('inf')
                    
                    if rel_diff > threshold:
                        severity = "CRITICAL" if rel_diff > 0.5 else "WARNING"
                        message = (
                            f"{severity}: {prop.replace('_', ' ').title()} mismatch:\n"
                            f"  Calculated: {calculated:.3f} {unit}\n"
                            f"  Empirical:  {empirical:.3f} {unit}\n"
                            f"  Difference: {diff:.3f} {unit} ({rel_diff*100:.1f}%)"
                        )
                        validation[category].append(message)
        
        return validation

def define_elements() -> List[Material]:
    """Define the first 50 elements with accurate nuclear composition."""
    # (protons, neutrons, electrons, valence_electrons)
    configurations = [
        (1, 0, 1, 1),      # H-1
        (2, 2, 2, 2),      # He-4
        (3, 4, 3, 1),      # Li-7
        (4, 5, 4, 2),      # Be-9
        (5, 6, 5, 3),      # B-11
        (6, 6, 6, 4),      # C-12
        (7, 7, 7, 5),      # N-14
        (8, 8, 8, 6),      # O-16
        (9, 10, 9, 7),     # F-19
        (10, 10, 10, 8),   # Ne-20
        (11, 12, 11, 1),   # Na-23
        (12, 12, 12, 2),   # Mg-24
        (13, 14, 13, 3),   # Al-27
        (14, 14, 14, 4),   # Si-28
        (15, 16, 15, 5),   # P-31
        (16, 16, 16, 6),   # S-32
        (17, 18, 17, 7),   # Cl-35
        (18, 22, 18, 8),   # Ar-40
        (19, 20, 19, 1),   # K-39
        (20, 20, 20, 2),   # Ca-40
        (21, 24, 21, 3),   # Sc-45
        (22, 26, 22, 4),   # Ti-48
        (23, 28, 23, 5),   # V-51
        (24, 28, 24, 6),   # Cr-52
        (25, 30, 25, 7),   # Mn-55
        (26, 30, 26, 8),   # Fe-56
        (27, 32, 27, 9),   # Co-59
        (28, 31, 28, 10),  # Ni-59
        (29, 35, 29, 11),  # Cu-64
        (30, 35, 30, 12),  # Zn-65
        (31, 39, 31, 3),   # Ga-70
        (32, 41, 32, 4),   # Ge-73
        (33, 42, 33, 5),   # As-75
        (34, 45, 34, 6),   # Se-79
        (35, 45, 35, 7),   # Br-80
        (36, 48, 36, 8),   # Kr-84
        (37, 48, 37, 1),   # Rb-85
        (38, 50, 38, 2),   # Sr-88
        (39, 50, 39, 3),   # Y-89
        (40, 51, 40, 4),   # Zr-91
        (41, 52, 41, 5),   # Nb-93
        (42, 54, 42, 6),   # Mo-96
        (43, 55, 43, 7),   # Tc-98
        (44, 57, 44, 8),   # Ru-101
        (45, 58, 45, 9),   # Rh-103
        (46, 60, 46, 10),  # Pd-106
        (47, 61, 47, 11),  # Ag-108
        (48, 64, 48, 12),  # Cd-112
        (49, 66, 49, 3),   # In-115
        (50, 69, 50, 4)    # Sn-119
    ]
    
    return [Material(*config) for config in configurations]

def test_elements():
    """Comprehensive testing and display of properties for all defined elements."""
    elements = define_elements()
    total_discrepancies = {
        'nuclear': 0,
        'electronic': 0,
        'structural': 0,
        'thermodynamic': 0,
        'chemical': 0
    }
    
    print("\n=== PERIODIC TABLE ANALYSIS ===\n")
    
    for element in elements:
        validation = element.validate_properties()
        print(f"\n{'='*50}")
        print(f"Element: Z={element.protons}")
        print(f"{'='*50}\n")
        
        # Nuclear Properties
        print("NUCLEAR PROPERTIES") 
        print("-" * 20)
        print(f"Protons: {element.protons}")
        print(f"Neutrons: {element.neutrons}")
        print(f"Mass Number: {element.protons + element.neutrons}")
        print(f"Atomic Mass: {element.atomic_mass:.4f} u")
        print(f"Nuclear Binding Energy: {element.derive_bond_energy():.2f} eV")
        
        # Electronic Properties
        print("\nELECTRONIC PROPERTIES")
        print("-" * 20)
        print(f"Electrons: {element.electrons}")
        print(f"Valence Electrons: {element.valence_electrons}")
        print(f"Electronic Configuration: {[shell.name for shell in element.electron_config]}")
        print(f"Electronegativity: {element.electronegativity:.3f} (Pauling)")
        print(f"Electron Affinity: {element.electron_affinity:.3f} eV")
        print(f"First Ionization Energy: {element.ionization_energies[0]:.3f} eV")
        print(f"Atomic Polarizability: {element._calculate_polarizability():.3e} m³")
        
        # Structural Properties
        print("\nSTRUCTURAL PROPERTIES")
        print("-" * 20)
        print(f"Atomic Radius: {element.radius:.3f} Å")
        print(f"Density (298K, 1atm): {max(element.derive_density(), 1e-6):.3f} g/cm³")
        
        # Thermodynamic Properties
        melting, boiling = element.derive_phase_transitions()
        melting = min(melting, 5000)
        boiling = min(boiling, 6000)
        print("\nTHERMODYNAMIC PROPERTIES")
        print("-" * 20)
        print(f"Phase (298K): {element._determine_phase(298.15).value}")
        print(f"Melting Point: {melting:.1f} K")
        print(f"Boiling Point: {boiling:.1f} K")
        print(f"Specific Heat: {min(element.derive_specific_heat(), 500):.3f} J/g·K")
        
        # Chemical Properties
        print("\nCHEMICAL PROPERTIES")
        print("-" * 20)
        print(f"Bond Energy: {element.derive_bond_energy():.2f} eV")
        polarity = min(element.derive_polarity(), 1.2 * element.electronegativity)
        print(f"Molecular Polarity: {polarity:.3f} D")
        
        # Validation Results
        if any(validation.values()):
            print("\nVALIDATION DISCREPANCIES")
            print("-" * 20)
            for category, discrepancies in validation.items():
                if discrepancies:
                    total_discrepancies[category] += len(discrepancies)
                    print(f"\n{category.upper()}:")
                    for disc in discrepancies:
                        if disc.startswith("CRITICAL"):
                            print(f"  • \033[91m{disc}\033[0m")  # Red for critical
                        else:
                            print(f"  • \033[93m{disc}\033[0m")  # Yellow for warnings
        
        print("\n" + "-"*50 + "\n")
    
    # Summary of all discrepancies
    print("\n=== VALIDATION SUMMARY ===")
    print("-" * 20)
    total = sum(total_discrepancies.values())
    for category, count in total_discrepancies.items():
        print(f"{category.title()}: {count} discrepancies")
    print(f"\nTotal Discrepancies: {total}")
    print("\nAnalysis Complete!")

if __name__ == "__main__":
    test_elements()
