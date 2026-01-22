import numpy as np
from ase.calculators.calculator import FileIOCalculator
from ase.calculators.calculator import SCFError
import ase.units

        The scratch directory, number of processor and threads as well as a few
        other command line options can be set using the arguments explained
        below. The remaining kwargs are copied as options to the input file.
        The calculator will convert these options to upper case
        (Q-Chem standard) when writing the input file.

        scratch: str
            path of the scratch directory
        np: int
            number of processors for the -np command line flag
        nt: int
            number of threads for the -nt command line flag
        pbs: boolean
            command line flag for pbs scheduler (see Q-Chem manual)
        basisfile: str
            path to file containing the basis. Use in combination with
            basis='gen' keyword argument.
        ecpfile: str
            path to file containing the effective core potential. Use in
            combination with ecp='gen' keyword argument.
        