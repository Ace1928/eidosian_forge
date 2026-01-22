import { describe, expect, it } from "vitest";
import { resolveCanvasMetrics } from "../src/core/canvas.js";

describe("canvas metrics", () => {
  it("scales dimensions with device pixel ratio", () => {
    const metrics = resolveCanvasMetrics(800, 600, 2);
    expect(metrics.cssWidth).toBe(800);
    expect(metrics.cssHeight).toBe(600);
    expect(metrics.pixelWidth).toBe(1600);
    expect(metrics.pixelHeight).toBe(1200);
    expect(metrics.ratio).toBe(2);
  });

  it("guards against invalid ratios", () => {
    const metrics = resolveCanvasMetrics(800, 600, 0);
    expect(metrics.ratio).toBe(1);
    expect(metrics.pixelWidth).toBe(800);
    expect(metrics.pixelHeight).toBe(600);
  });
});
