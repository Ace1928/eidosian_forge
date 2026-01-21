import { expect, test, type Page } from "@playwright/test";

const setupDeterministicFrame = async (page: Page) => {
  await page.addInitScript(() => {
    let frame = 0;
    const fixedNow = 0;
    performance.now = () => fixedNow;
    window.requestAnimationFrame = (callback) => {
      frame += 1;
      if (frame <= 1) {
        window.setTimeout(() => callback(fixedNow), 0);
      }
      return frame;
    };
    window.cancelAnimationFrame = () => undefined;
  });
};

test.describe("visual regression", () => {
  test("renders deterministic canvas and HUD", async ({ page }) => {
    await setupDeterministicFrame(page);
    await page.goto("/");
    await page.waitForFunction(() => {
      const text = document.getElementById("resource-info")?.textContent ?? "";
      return text.includes("Tick");
    });
    await page.evaluate(() => document.fonts.ready);

    const canvas = page.locator("#game");
    await expect(canvas).toBeVisible();
    await expect(canvas).toHaveScreenshot("canvas.png", { animations: "disabled" });

    const hud = page.locator("#hud");
    await expect(hud).toBeVisible();
    await expect(hud).toHaveScreenshot("hud.png", { animations: "disabled" });
  });
});
