from tensorflow.python.util import nest_util
def is_nested(structure):
    return nest_util.is_nested(nest_util.Modality.DATA, structure)