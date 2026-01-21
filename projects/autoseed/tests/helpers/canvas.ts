export const installCanvasContextStub = () => {
  const original = HTMLCanvasElement.prototype.getContext;

  const createGradient = () => ({
    addColorStop: () => undefined
  });

  const createContext = (canvas: HTMLCanvasElement) => ({
    canvas,
    setTransform: () => undefined,
    createLinearGradient: () => createGradient(),
    fillRect: () => undefined,
    save: () => undefined,
    restore: () => undefined,
    beginPath: () => undefined,
    ellipse: () => undefined,
    arc: () => undefined,
    fill: () => undefined,
    stroke: () => undefined,
    translate: () => undefined,
    scale: () => undefined,
    fillText: () => undefined,
    set globalAlpha(value: number) {
      void value;
    },
    set fillStyle(value: string) {
      void value;
    },
    set strokeStyle(value: string) {
      void value;
    },
    set lineWidth(value: number) {
      void value;
    },
    set font(value: string) {
      void value;
    }
  });

  HTMLCanvasElement.prototype.getContext = function getContext(type: string) {
    if (type !== "2d") {
      return null;
    }
    return createContext(this);
  };

  return () => {
    HTMLCanvasElement.prototype.getContext = original;
  };
};
