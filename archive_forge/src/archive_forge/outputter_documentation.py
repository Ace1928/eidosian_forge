from abc import ABC, abstractmethod
from fugue.dataframe import DataFrames
from fugue.extensions.context import ExtensionContext
Process the collection of dataframes on driver side

        .. note::

          * It runs on driver side
          * The dataframes are not necessarily local, for example a SparkDataFrame
          * It is engine aware, you can put platform dependent code in it (for example
            native pyspark code) but by doing so your code may not be portable. If you
            only use the functions of the general ExecutionEngine, it's still portable.

        :param dfs: dataframe collection to process
        