# 🔐 MKey Forge Installation

Key management and cryptographic utilities.

## Quick Install

```bash
pip install -e ./mkey_forge
python -c "from mkey_forge import KeyManager; print('✓ Ready')"
```

## CLI Usage

```bash
mkey-forge generate --type rsa --bits 4096
mkey-forge encrypt file.txt --key keyfile
mkey-forge --help
```

