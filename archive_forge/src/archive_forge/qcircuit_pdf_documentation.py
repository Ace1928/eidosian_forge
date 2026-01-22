import errno
import os
from pylatex import Document, NoEscape, Package
from cirq import circuits
from cirq.contrib.qcircuit.qcircuit_diagram import circuit_to_latex_using_qcircuit
Compiles the QCircuit-based latex diagram of the given circuit.

    Args:
        circuit: The circuit to produce a pdf of.
        filepath: Where to output the pdf.
        pdf_kwargs: The arguments to pass to generate_pdf.
        qcircuit_kwargs: The arguments to pass to
            circuit_to_latex_using_qcircuit.
        clean_ext: The file extensions to clean up after compilation. By
            default, latexmk is used with the '-pdfps' flag, which produces
            intermediary dvi and ps files.
        documentclass: The documentclass of the latex file.

    Raises:
        OSError, IOError: If cleanup fails.
    