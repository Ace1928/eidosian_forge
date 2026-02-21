# TODO: gis_forge

## 🚨 High Priority
- [x] **Cleanup**: Remove `eidos_venv/` from git tracking.
- [x] **Consolidate**: Moved `global_info.py` to `gis_forge.defaults`.

## 🟡 Medium Priority
- [x] **Feature**: Support YAML/TOML in addition to JSON.
- [x] **Validation**: Integrate `pydantic` schemas for config validation.

## 🟢 Low Priority
- [ ] **Etcd/Consul**: Support distributed key-value stores for multi-node deployments.

## ci-hardening
- [ ] Normalize figlet_forge compat/encoding_adjuster.py structure (currently duplicate module-level helper definitions) after CI stabilization to reduce maintenance drift and line-number ambiguity in failure triage.

## code_forge_eval_os
- [x] Implement evaluation/observability operating system core: task contracts, run matrix, trace schema, replay store, staleness metrics, and CLI wiring.
- [x] Add production docs + references for SWE-bench, OpenTelemetry trace model, W3C PROV, HTTP stale-while-revalidate semantics.
- [x] Add and pass code_forge test suite coverage for new eval_os modules and CLI command behavior.
