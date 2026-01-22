import argparse
import os
import nibabel as nib
from nibabel.orientations import aff2axcodes
from nibabel.streamlines import Field

Convert tractograms (TCK -> TRK).
