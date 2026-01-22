import collections
import numpy as np
import tensorflow.compat.v2 as tf
def validate_accumulator_extract_and_restore(self, combiner, data, expected):
    """Validate that the extract<->restore loop loses no data."""
    acc = combiner.compute(data)
    extracted_data = combiner.extract(acc)
    restored_acc = combiner.restore(extracted_data)
    self.assert_extracted_output_equal(combiner, acc, restored_acc)
    self.assertAllCloseOrEqual(expected, combiner.extract(restored_acc))