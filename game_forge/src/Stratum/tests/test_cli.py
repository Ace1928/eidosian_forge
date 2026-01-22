import io
from contextlib import redirect_stdout
from unittest import mock

from scenarios import cli


def test_cli_list_outputs_scenarios():
    buf = io.StringIO()
    with redirect_stdout(buf):
        cli.main(["--list"])
    out = buf.getvalue()
    assert "collapse" in out
    assert "runtime" in out
    assert "screensaver" in out


def test_cli_collapse_invokes_runner():
    with mock.patch("scenarios.cli._run_collapse") as runner:
        cli.main(["--scenario", "collapse", "--grid", "4", "--ticks", "2", "--microticks", "1", "--output", "./out"])
        runner.assert_called_once()


def test_cli_runtime_invokes_runner():
    with mock.patch("scenarios.cli._run_runtime") as runner:
        cli.main(["--scenario", "runtime", "--grid", "4", "--runtime", "1", "--microticks", "1", "--snapshot", "1"])
        runner.assert_called_once()


def test_cli_screensaver_invokes_runner():
    with mock.patch("scenarios.cli._run_screensaver") as runner:
        cli.main(["--scenario", "screensaver", "--grid", "0", "--microticks", "1", "--fps", "10"])
        runner.assert_called_once()
