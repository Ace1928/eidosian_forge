# Tutorial: Code Forge Roundtrip on a Sample Project

## 1. Prepare
```bash
cd /data/data/com.termux/files/home/eidosian_forge
source eidosian_venv/bin/activate
```

## 2. Digest + Integrate + Reconstruct
```bash
code-forge roundtrip sms_forge \
  --workspace-dir data/code_forge/roundtrip/sms_forge_tutorial \
  --mode analysis \
  --sync-knowledge \
  --export-graphrag \
  --integration-policy effective_run
```

## 3. Validate Contracts
```bash
code-forge validate-roundtrip \
  --workspace-dir data/code_forge/roundtrip/sms_forge_tutorial \
  --verify-hashes
```

## 4. Preview Apply (No Changes)
```bash
code-forge apply-reconstruction \
  --reconstructed-root data/code_forge/roundtrip/sms_forge_tutorial/reconstructed \
  --target-root sms_forge \
  --backup-root Backups/code_forge_roundtrip \
  --require-manifest \
  --dry-run
```

## 5. Apply (Promotion)
```bash
code-forge apply-reconstruction \
  --reconstructed-root data/code_forge/roundtrip/sms_forge_tutorial/reconstructed \
  --target-root sms_forge \
  --backup-root Backups/code_forge_roundtrip \
  --require-manifest
```

## 6. Query Provenance via MCP
Use MCP tool:
- `code_forge_provenance`

Recommended args:
- `{"root_path": "/data/data/com.termux/files/home/eidosian_forge/sms_forge", "stage": "roundtrip", "limit": 5}`

## Outputs to Inspect
- `data/code_forge/roundtrip/sms_forge_tutorial/roundtrip_summary.json`
- `data/code_forge/roundtrip/sms_forge_tutorial/provenance_links.json`
- `data/code_forge/roundtrip/sms_forge_tutorial/reconstructed/reconstruction_manifest.json`
- `data/code_forge/roundtrip/sms_forge_tutorial/parity_report.json`
