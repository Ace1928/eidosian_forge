import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "tests/visual",
  retries: 0,
  use: {
    baseURL: "http://127.0.0.1:8080",
    viewport: { width: 1280, height: 720 },
    deviceScaleFactor: 1
  },
  webServer: {
    command: "npm run build && node scripts/serve.mjs",
    port: 8080,
    reuseExistingServer: false,
    timeout: 120000
  }
});
