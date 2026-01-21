// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { installCanvasContextStub } from "./helpers/canvas.js";
import { loadHudMarkup } from "./helpers/dom.js";

describe("app bootstrap", () => {
  let restoreCanvas: (() => void) | null = null;

  beforeEach(() => {
    vi.resetModules();
    vi.clearAllMocks();
    Object.defineProperty(window, "innerWidth", { value: 800, configurable: true });
    Object.defineProperty(window, "innerHeight", { value: 600, configurable: true });
    Object.defineProperty(window, "devicePixelRatio", { value: 1, configurable: true });
    vi.stubGlobal(
      "requestAnimationFrame",
      vi.fn(() => 1)
    );
    vi.stubGlobal("cancelAnimationFrame", vi.fn());
    restoreCanvas = installCanvasContextStub();
  });

  afterEach(() => {
    restoreCanvas?.();
    vi.unstubAllGlobals();
  });

  it("starts the engine when the canvas exists", async () => {
    loadHudMarkup();

    await import("../src/app.js");

    const canvas = document.querySelector<HTMLCanvasElement>("#game");
    expect(canvas?.width).toBe(800);
    expect(canvas?.height).toBe(600);
    expect(window.requestAnimationFrame).toHaveBeenCalledTimes(1);
  });

  it("throws when the canvas is missing", async () => {
    document.body.innerHTML = `
      <div id="resource-info"></div>
      <div id="system-info"></div>
      <div id="body-info"></div>
      <div id="build-info"></div>
      <div id="tech-info"></div>
      <div id="design-info"></div>
    `;

    await expect(import("../src/app.js")).rejects.toThrow("Canvas element not found");
    expect(window.requestAnimationFrame).not.toHaveBeenCalled();
  });
});
