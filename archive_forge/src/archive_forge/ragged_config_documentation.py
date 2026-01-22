Whether incompatible row-partitioning dtypes should be auto-converted.

  If true, then operations that combine RaggedTensors but have different
  row-partitioning tensor dtypes will be automatically cast to a
  compatible dtype (`tf.int64`).  If false, then such operations will result
  in an error.

  Returns:
    `bool`
  