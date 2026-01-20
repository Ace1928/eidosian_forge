import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    include: ["tests/integration/**/*.test.ts"],
    coverage: {
      provider: "v8",
      enabled: false
    }
  }
});
