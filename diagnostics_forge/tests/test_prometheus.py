from __future__ import annotations

from diagnostics_forge import DiagnosticsForge
from diagnostics_forge.cli import main


def test_prometheus_text_contains_core_metrics(tmp_path):
    diag = DiagnosticsForge(log_dir=tmp_path, service_name="prom_test")
    timer_id = diag.start_timer("phase_a")
    diag.stop_timer(timer_id)

    text = diag.get_prometheus_metrics()

    assert "# HELP eidos_cpu_percent" in text
    assert 'eidos_cpu_percent{service="prom_test"}' in text
    assert 'eidos_metric_count{service="prom_test",metric="phase_a"} 1' in text


def test_prometheus_cli_outputs_metrics(capsys):
    code = main(["prometheus"])
    out = capsys.readouterr().out
    assert code == 0
    assert "eidos_cpu_percent" in out
