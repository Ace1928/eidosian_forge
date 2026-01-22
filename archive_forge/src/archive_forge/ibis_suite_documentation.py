from unittest import TestCase
import ibis
from fugue import ExecutionEngine, FugueWorkflow, register_default_sql_engine
from fugue_ibis import IbisEngine, as_fugue, as_ibis, run_ibis
Ibis test suite.
    Any new engine from :class:`~fugue_ibis.execution.ibis_engine.IbisEngine`
    should also pass this test suite.
    