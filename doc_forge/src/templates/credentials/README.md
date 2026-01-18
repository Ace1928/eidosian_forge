# ⚠️ EIDOSIAN CREDENTIAL SECURITY PROTOCOL ⚠️

## ZERO-TOLERANCE SECURITY POLICY

Templates in this directory contain **PLACEHOLDER VALUES ONLY**.
Real credentials must **NEVER** appear in version control.

## CREDENTIAL DEPLOYMENT PROCEDURE:

1. Copy templates to secure locations **OUTSIDE** this repository
   - PyPI credentials: `~/.pypirc`
   - Environment variables: `~/.env` or through your CI system
   - API keys: Secure credential store only

2. Replace placeholders with real values in the copied files

3. **NEVER** commit files containing real credentials

## AUTOMATIC SECURITY MEASURES:

- Repository-level credential exclusion (.gitignore)
- Global Git exclusion patterns
- Automatic untracking of detected credential files

If you discover committed credentials, revoke them **IMMEDIATELY**
and rewrite Git history to remove all traces.
