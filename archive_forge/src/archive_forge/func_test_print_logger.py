import cirq
import cirq_google as cg
from cirq_google.workflow.progress import _PrintLogger
def test_print_logger(capsys):
    pl = _PrintLogger(n_total=10)
    shared_rt_info = cg.SharedRuntimeInfo(run_id='hi mom')
    pl.initialize()
    for i in range(10):
        exe_result = cg.ExecutableResult(spec=None, runtime_info=cg.RuntimeInfo(execution_index=i), raw_data=cirq.ResultDict(params=cirq.ParamResolver({}), measurements={}))
        pl.consume_result(exe_result, shared_rt_info)
    pl.finalize()
    assert capsys.readouterr().out == '\n\r1 / 10\r2 / 10\r3 / 10\r4 / 10\r5 / 10\r6 / 10\r7 / 10\r8 / 10\r9 / 10\r10 / 10\n'