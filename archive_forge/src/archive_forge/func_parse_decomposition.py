from . import processMagmaFile
from . import processRurFile
from . import processComponents
def parse_decomposition(text):
    if processMagmaFile.contains_magma_output(text):
        return processMagmaFile.decomposition_from_magma(text)
    if processRurFile.contains_rur(text):
        return processRurFile.decomposition_from_rur(text)
    if processComponents.contains_ideal_components(text):
        return processComponents.decomposition_from_components(text)
    raise Exception('solution file format not recognized')