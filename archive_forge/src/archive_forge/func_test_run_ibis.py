from unittest import TestCase
import ibis
from fugue import ExecutionEngine, FugueWorkflow, register_default_sql_engine
from fugue_ibis import IbisEngine, as_fugue, as_ibis, run_ibis
def test_run_ibis(self):

    def _test1(con: ibis.BaseBackend) -> ibis.Expr:
        tb = con.table('a')
        return tb

    def _test2(con: ibis.BaseBackend) -> ibis.Expr:
        tb = con.table('a')
        return tb.mutate(c=tb.a + tb.b)
    dag = FugueWorkflow()
    df = dag.df([[0, 1], [2, 3]], 'a:long,b:long')
    res = run_ibis(_test1, ibis_engine=self.ibis_engine, a=df)
    res.assert_eq(df)
    df = dag.df([[0, 1], [2, 3]], 'a:long,b:long')
    res = run_ibis(_test2, ibis_engine=self.ibis_engine, a=df)
    df2 = dag.df([[0, 1, 1], [2, 3, 5]], 'a:long,b:long,c:long')
    res.assert_eq(df2)
    dag.run(self.engine)