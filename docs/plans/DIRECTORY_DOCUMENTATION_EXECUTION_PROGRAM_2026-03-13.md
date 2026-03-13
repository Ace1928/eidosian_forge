# Directory Documentation Execution Program

Date: 2026-03-13
Status: Active subordinate execution slice
Parent program: `docs/plans/EIDOSIAN_MASTER_IMPLEMENTATION_PROGRAM_2026-03-07.md`

## Intent

Treat directory documentation as a maintained system rather than ad hoc markdown by:
- generating managed README files from repository facts,
- exposing programmatic coverage/render/diff/upsert APIs,
- keeping references and route claims conservative and testable,
- and making coverage measurable across the documented forge estate.

## Completed

- [x] Added directory inventory and README rendering contract in `doc_forge/src/doc_forge/scribe/directory_docs.py`
- [x] Added documentation scope policy in `cfg/documentation_policy.json`
- [x] Added batch README generation CLI in `scripts/generate_directory_readmes.py`
- [x] Added Doc Forge service APIs for:
  - coverage
  - readme lookup
  - render
  - diff
  - single-directory upsert
  - batch upsert
- [x] Added Atlas-side documentation control APIs and operator actions for:
  - coverage refresh
  - coverage lookup
  - render
  - diff
  - upsert
- [x] Fixed descendant-write behavior so selected roots generate README files for nested directories
- [x] Fixed generated ancestor/forge README reference paths
- [x] Expanded route detection beyond `@app.*` to include router-style path decorators
- [x] Added validating-test reference detection to generated directory READMEs
- [x] Fed documentation coverage status into scheduler/runtime/autonomy surfaces
- [x] Captured and saved documentation-system references under `docs/external_references/2026-03-13-directory-doc-system/`
- [x] Generated managed README coverage for the current documented-prefix estate
- [x] Produced post-wave coverage artifact:
  - `reports/docs/directory_docs_postwave_all.json`
  - `required_directory_count=2364`
  - `missing_readme_count=0`

## Current Guarantees

- Managed README claims are derived from:
  - tracked repository structure
  - conservative route scanning
  - local file presence
  - explicit policy overrides
- Documentation APIs are directly callable through Doc Forge service routes.
- Batch generation is root-scoped and repeatable.

## Open Work

- [~] Add Atlas control-plane views for documentation coverage, diffing, and batch regeneration.
- [~] Add source/test linking beyond directory-local detection so package roots can reference sibling test suites accurately.
- [~] Add coverage drift reporting to scheduler/autonomy/runtime artifacts.
- [ ] Add managed suppression/override contracts for intentionally undocumented directories.
- [ ] Add review gates for generated README quality on high-risk directories before promotion.

## Validation

- [x] `doc_forge/src/doc_forge/scribe/tests/test_directory_docs.py`
- [x] `doc_forge/src/doc_forge/scribe/tests/test_service.py`
- [x] `scripts/tests/test_generate_directory_readmes.py`

## Next Execution Slice

1. expose documentation inventory and diff controls in Atlas
2. add batch-regeneration and directory-doc navigation UI in Atlas
3. improve validation-source precision and suppress low-value generated or archived test references
