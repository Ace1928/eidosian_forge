import warnings
from Bio import BiopythonWarning
from Bio import MissingPythonDependencyError
from Bio.motifs import jaspar, matrix
def fetch_motifs_by_name(self, name):
    """Fetch a list of JASPAR motifs from a JASPAR DB by the given TF name(s).

        Arguments:
        name - a single name or list of names
        Returns:
        A list of Bio.motifs.jaspar.Motif objects

        Notes:
        Names are not guaranteed to be unique. There may be more than one
        motif with the same name. Therefore even if name specifies a single
        name, a list of motifs is returned. This just calls
        self.fetch_motifs(collection = None, tf_name = name).

        This behaviour is different from the TFBS perl module's
        get_Matrix_by_name() method which always returns a single matrix,
        issuing a warning message and returning the first matrix retrieved
        in the case where multiple matrices have the same name.

        """
    return self.fetch_motifs(collection=None, tf_name=name)