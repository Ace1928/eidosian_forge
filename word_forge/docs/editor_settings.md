# Recommended Editor Settings

The repository does not include project level `.code-workspace` files.
Add these settings to your personal VS Code workspace or `settings.json` if you
would like to replicate the original configuration.

```jsonc
{
  "github.copilot.chat.agent.thinkingTool": true,
  "github.copilot.chat.codesearch.enabled": true,
  "terminal.integrated.tabs.defaultColor": "terminal.ansiBlue",
  "chat.agent.maxRequests": 50
}
```

These values enable Copilot Chat integration and set the default terminal tab
color. They are optional and can be adjusted to suit your environment.
