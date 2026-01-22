import os
import pickle
from traits.api import Float
from traits import __version__
def generate_pickles():
    pickle_directory = os.path.abspath('.')
    for protocol in SUPPORTED_PICKLE_PROTOCOLS:
        for description, picklee in PICKLEES.items():
            write_pickle_file(description, picklee, protocol, pickle_directory)