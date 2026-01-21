import { describe, expect, it } from "vitest";
import { RNG } from "../src/core/random.js";

describe("RNG", () => {
  it("produces deterministic sequences", () => {
    const rngA = new RNG(123);
    const rngB = new RNG(123);
    const sequenceA = [rngA.next(), rngA.next(), rngA.next()];
    const sequenceB = [rngB.next(), rngB.next(), rngB.next()];
    expect(sequenceA).toEqual(sequenceB);
  });

  it("picks in range", () => {
    const rng = new RNG(9);
    const value = rng.int(2, 4);
    expect(value).toBeGreaterThanOrEqual(2);
    expect(value).toBeLessThanOrEqual(4);
  });

  it("supports float ranges", () => {
    const rng = new RNG(5);
    const value = rng.float(1, 1.5);
    expect(value).toBeGreaterThanOrEqual(1);
    expect(value).toBeLessThanOrEqual(1.5);
  });

  it("throws on invalid ranges", () => {
    const rng = new RNG(2);
    expect(() => rng.int(5, 2)).toThrow("max must be >= min");
    expect(() => rng.float(5, 2)).toThrow("max must be >= min");
  });

  it("picks and weighted picks values", () => {
    const rng = new RNG(3);
    const pick = rng.pick(["a", "b", "c"]);
    expect(["a", "b", "c"]).toContain(pick);
    const weighted = rng.weightedPick(["x", "y"], [0, 1]);
    expect(weighted).toBe("y");
  });

  it("throws on empty pick and invalid weight arrays", () => {
    const rng = new RNG(3);
    expect(() => rng.pick([])).toThrow("Cannot pick from empty array");
    expect(() => rng.weightedPick(["a"], [])).toThrow(
      "Items and weights must be same length and non-empty"
    );
  });

  it("throws when pick hits an undefined slot", () => {
    const rng = new RNG(4);
    const items = new Array(1) as string[];
    expect(() => rng.pick(items)).toThrow("Random selection failed");
  });

  it("falls back to the last weighted item when roll never settles", () => {
    const rng = new RNG(5);
    const value = rng.weightedPick(["a", "b"], [Number.NaN, Number.NaN]);
    expect(value).toBe("b");
  });
});
