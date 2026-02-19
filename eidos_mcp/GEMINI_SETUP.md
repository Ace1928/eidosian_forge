# Gemini CLI Setup for Eidos MCP

This server is configured for Gemini CLI over **Streamable HTTP** at `http://127.0.0.1:8928/mcp`.

## 1. Connection Methods

Your `~/.gemini/settings.json` supports these connections:

| Server ID | Transport | Auth | Description |
| :--- | :--- | :--- | :--- |
| **eidosian_nexus** | Streamable HTTP | None | **Default/Local**. Connects to `http://127.0.0.1:8928/mcp`. |
| **eidosian_nexus_google** | Streamable HTTP | **Google ADC** | Same endpoint with Google bearer token if strict auth is enabled. |

Recommended `~/.gemini/settings.json` entry:

```json
{
  "mcpServers": {
    "eidosian_nexus": {
      "httpUrl": "http://127.0.0.1:8928/mcp",
      "url": "http://127.0.0.1:8928/mcp",
      "trust": true
    }
  }
}
```

## 2. Running the MCP Service

To use `eidosian_nexus` or `eidosian_nexus_google`, the server must be running.

### Manual Start
```bash
./run_server.sh
```

### Systemd Service (Recommended)
1.  **Check Status**: `systemctl --user status eidos-mcp`
2.  **Logs**: `journalctl --user -u eidos-mcp -f`
3.  **Install/Update**:
    ```bash
    mkdir -p ~/.config/systemd/user
    cp systemd/eidos-mcp.service ~/.config/systemd/user/
    systemctl --user daemon-reload
    systemctl --user enable --now eidos-mcp.service
    ```

## 3. Google Authentication Setup

To use **eidosian_nexus_google**, you must have Google Application Default Credentials (ADC) set up on this machine. Gemini uses these to mint tokens.

### Prerequisites
1.  **gcloud CLI**: Install the Google Cloud SDK.
    - [Installation Guide](https://cloud.google.com/sdk/docs/install)
2.  **Login**:
    ```bash
    gcloud auth application-default login
    ```
    This creates `~/.config/gcloud/application_default_credentials.json`.

### How it Works
1.  Gemini detects `authProviderType: "google_credentials"`.
2.  It loads your ADC.
3.  It adds an `Authorization: Bearer <token>` header to MCP HTTP requests.
4.  The Eidos server receives this token (currently in `open` mode, but ready for validation).

### Token Persistence
OAuth tokens are cached in `~/.gemini/mcp-oauth-tokens.json` to prevent re-authentication loops.

## Verification
1.  Switch to `eidosian_nexus_google` in Gemini.
2.  Run `/mcp` to verify connection.
3.  Run `auth_whoami` tool to see connection details.
