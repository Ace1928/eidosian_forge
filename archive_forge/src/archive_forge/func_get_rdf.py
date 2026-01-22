from ase.neighborlist import build_neighbor_list, get_distance_matrix, get_distance_indices
from ase.ga.utilities import get_rdf
from ase import Atoms
def get_rdf(self, rmax, nbins, imageIdx=None, elements=None, return_dists=False):
    """Get RDF.

        Wrapper for :meth:`ase.ga.utilities.get_rdf` with more selection possibilities.

        Parameters:

        rmax: float
            Maximum distance of RDF.
        nbins: int
            Number of bins to divide RDF.
        imageIdx: int/slice/None
            Images to analyze, see :func:`_get_slice` for details.
        elements: str/int/list/tuple
            Make partial RDFs.

        If elements is *None*, a full RDF is calculated. If elements is an *integer* or a *list/tuple
        of integers*, only those atoms will contribute to the RDF (like a mask). If elements
        is a *string* or a *list/tuple of strings*, only Atoms of those elements will contribute.

        Returns:

        return: list of lists / list of tuples of lists
            If return_dists is True, the returned tuples contain (rdf, distances). Otherwise
            only rdfs for each image are returned.
        """
    sl = self._get_slice(imageIdx)
    r = []
    el = None
    for image in self.images[sl]:
        if elements is None:
            tmpImage = image
        elif isinstance(elements, int):
            tmpImage = Atoms(cell=image.get_cell(), pbc=image.get_pbc())
            tmpImage.append(image[elements])
        elif isinstance(elements, str):
            tmpImage = Atoms(cell=image.get_cell(), pbc=image.get_pbc())
            for idx in self._get_symbol_idxs(image, elements):
                tmpImage.append(image[idx])
        elif isinstance(elements, list) or isinstance(elements, tuple):
            if all((isinstance(x, int) for x in elements)):
                if len(elements) == 2:
                    el = elements
                    tmpImage = image
                else:
                    tmpImage = Atoms(cell=image.get_cell(), pbc=image.get_pbc())
                    for idx in elements:
                        tmpImage.append(image[idx])
            elif all((isinstance(x, str) for x in elements)):
                tmpImage = Atoms(cell=image.get_cell(), pbc=image.get_pbc())
                for element in elements:
                    for idx in self._get_symbol_idxs(image, element):
                        tmpImage.append(image[idx])
            else:
                raise ValueError('Unsupported type of elements given in ase.geometry.analysis.Analysis.get_rdf!')
        else:
            raise ValueError('Unsupported type of elements given in ase.geometry.analysis.Analysis.get_rdf!')
        r.append(get_rdf(tmpImage, rmax, nbins, elements=el, no_dists=not return_dists))
    return r