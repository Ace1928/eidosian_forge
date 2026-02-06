# Moltbook Sources

Date: 2026-02-06

Official references captured for traceability and rollback.

## Core Site
- https://www.moltbook.com/

## Policies
- https://www.moltbook.com/terms (Last updated: January 2026)
- https://www.moltbook.com/privacy (Last updated: January 2026)

## Developer Platform
- https://www.moltbook.com/developers
  - App keys prefixed `moltdev_`
  - Identity token verification endpoint `POST /api/v1/agents/verify-identity`
  - Token generation endpoint `POST /api/v1/agents/me/identity-token`
- https://www.moltbook.com/developers/apply
- https://moltbook.com/developers.md (linked from developer page)

## Skills and Messaging (curl -sL)
- https://www.moltbook.com/skill.md (saved to `docs/moltbook_skill.md`, sanitized)
- https://www.moltbook.com/heartbeat.md (saved to `docs/moltbook_heartbeat.md`, sanitized)
- https://www.moltbook.com/messaging.md (saved to `docs/moltbook_messaging.md`, sanitized)
- https://www.moltbook.com/skill.json (metadata)

## Browsing Endpoints
- https://www.moltbook.com/u (agents list)
- https://www.moltbook.com/m (communities list)

Notes
- Use the www domain for API and docs to avoid auth header stripping on redirect.
- Skill, heartbeat, and messaging docs were fetched via curl and sanitized to ASCII for repo storage.

## External References Used in Comments (2026-02-06)
- https://slsa.dev/
- https://openssf.org/projects/sigstore/
- https://www.cisa.gov/sbom
- https://docs.npmjs.com/viewing-package-provenance
- https://www.hhs.gov/answers/hipaa/what-is-phi/index.html
- https://www.hhs.gov/hipaa/for-professionals/privacy/index.html
- https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html
- https://consumer.ftc.gov/consumer-alerts/2024/07/can-you-spot-investment-scam
- https://www.finra.org/investors/insights/investment-group-imposter-scams
- https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-alerts/beware-fraudsters-impersonating-investment-professionals-and-firms-investor-alert

## Sharding References (research)
- https://github.com/olric-data/olric
- https://github.com/microsoft/garnet
- https://github.com/openxla/shardy
