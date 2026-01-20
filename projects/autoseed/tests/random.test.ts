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
});
