@classmethod
def from_model_path(cls, model_path):
    """Creates a processor using the given model path.

    Args:
      model_path: The path to the stored model.

    Returns:
      An instance implementing this Processor class.
    """
    raise NotImplementedError()