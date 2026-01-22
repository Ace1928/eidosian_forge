import abc
from tensorflow.python.util.tf_export import tf_export
@abc.abstractmethod
def transform_feature(self, transformation_cache, state_manager):
    """Returns intermediate representation (usually a `Tensor`).

    Uses `transformation_cache` to create an intermediate representation
    (usually a `Tensor`) that other feature columns can use.

    Example usage of `transformation_cache`:
    Let's say a Feature column depends on raw feature ('raw') and another
    `FeatureColumn` (input_fc). To access corresponding `Tensor`s,
    transformation_cache will be used as follows:

    ```python
    raw_tensor = transformation_cache.get('raw', state_manager)
    fc_tensor = transformation_cache.get(input_fc, state_manager)
    ```

    Args:
      transformation_cache: A `FeatureTransformationCache` object to access
        features.
      state_manager: A `StateManager` to create / access resources such as
        lookup tables.

    Returns:
      Transformed feature `Tensor`.
    """
    pass