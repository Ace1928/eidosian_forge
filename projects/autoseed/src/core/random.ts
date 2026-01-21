export class RNG {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  next(): number {
    // Mulberry32 PRNG
    let t = (this.state += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    const result = ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    this.state = t >>> 0;
    return result;
  }

  int(min: number, max: number): number {
    if (max < min) {
      throw new Error("max must be >= min");
    }
    const span = max - min + 1;
    return Math.floor(this.next() * span) + min;
  }

  float(min: number, max: number): number {
    if (max < min) {
      throw new Error("max must be >= min");
    }
    return this.next() * (max - min) + min;
  }

  pick<T>(items: readonly T[]): T {
    if (items.length === 0) {
      throw new Error("Cannot pick from empty array");
    }
    const choice = items[this.int(0, items.length - 1)];
    if (choice === undefined) {
      throw new Error("Random selection failed");
    }
    return choice;
  }

  weightedPick<T>(items: readonly T[], weights: readonly number[]): T {
    if (items.length !== weights.length || items.length === 0) {
      throw new Error("Items and weights must be same length and non-empty");
    }
    const total = weights.reduce((sum, value) => sum + value, 0);
    let roll = this.next() * total;
    for (let i = 0; i < items.length; i += 1) {
      roll -= weights[i]!;
      if (roll <= 0) {
        return items[i] as T;
      }
    }
    return items[items.length - 1] as T;
  }
}
