from io import StringIO
import ase.io
from .parse_lammps_data_file import lammpsdata_file_extracted_sections
from .comparison import compare_with_pytest_approx

Create an atoms object and write it to a lammps data file
