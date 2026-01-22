def threshold_fpr(self, fpr):
    """Approximate the log-odds threshold which makes the type I error (false positive rate)."""
    i = self.n_points
    prob = 0.0
    while prob < fpr:
        i -= 1
        prob += self.bg_density[i]
    return self.min_score + i * self.step