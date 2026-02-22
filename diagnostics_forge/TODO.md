# TODO: diagnostics_forge

## 🚨 High Priority
- [x] **Persist**: Auto-save metrics on exit (via `atexit`).
- [x] **Rotation**: Verified log rotation works (`tests/test_diag.py`).

## 🟡 Medium Priority
- [x] **Format**: Added JSON logging option.
- [x] ~~**Dashboard**: Create a simple CLI dashboard to view metrics json.~~ (Implemented `diagnostics-forge dashboard --metrics-file ...`)

## 🟢 Low Priority
- [x] ~~**Prometheus**: Add an exporter endpoint.~~ (`diagnostics-forge prometheus` + `--serve` endpoint on `/metrics`)
