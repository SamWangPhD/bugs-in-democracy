import os, random
from pathlib import Path
from collections import defaultdict
from itertools import permutations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from rcv_distribution import *
from MDS_analysis import *
from voting_rules import *

def normalize_data(a):
    a = np.asarray(a, float).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0: return a
    amin, amax = a.min(), a.max()
    return (a - amin) / (amax - amin) if amax > amin else np.zeros_like(a)


def sample_from(source, n, lo=0.0, hi=1.0, batch=5000):
    """Draw n samples in [lo,hi] either from a KDE (has .resample) or from an array (empirical)."""
    out = []
    while len(out) < n:
        k = max(batch, n - len(out))
        if hasattr(source, "resample"):      # KDE
            draw = source.resample(k).ravel()
        else:                                # raw data -> empirical resampling
            arr = np.asarray(source, float).ravel()
            draw = np.random.choice(arr, size=k, replace=True)
        draw = draw[(draw >= lo) & (draw <= hi)]
        out.extend(draw.tolist())
    return np.asarray(out[:n])


def generate_and_group_permutations(items, choices):
    grouped = defaultdict(list)
    for i in range(1, choices + 1):
        for perm in permutations(items, i):
            grouped[perm[0]].append(perm)
    return grouped

# ---- main function (expects voter_kde OR array; candidate_dist is (x, y)) ----
def top_n_gamma(n, x_voter_plot, y_voter_plot, voter_dist, candidate_dist, visualize=False):
    x_can, y_can = candidate_dist

    # sample candidate points proportional to candidate density
    probs = np.clip(np.asarray(y_can, float), 0, None)
    probs = probs / probs.sum() if probs.sum() > 0 else np.ones_like(probs) / probs.size
    k = min(10, x_can.size)
    replace = x_can.size < 10
    candidate_points = np.random.choice(x_can, size=k, p=probs, replace=replace)

    # gamma -> number of consistent voters (0..1000)
    df_gamma = pd.read_csv("election_table(2).csv")
    gamma_vals = df_gamma.loc[df_gamma["candidates"].eq(n), "gamma"].to_numpy()
    random_gamma = float(np.random.choice(gamma_vals)) if gamma_vals.size else 0.5
    consistent_voters = int(np.clip(round(random_gamma * 1000), 0, 1000))

    # round 1: choose top-n by nearest-candidate counts from 1000 samples
    sampled_points_primary = sample_from(voter_dist, 1000, 0.0, 1.0)
    counts = np.zeros(candidate_points.size, dtype=int)
    for p in sampled_points_primary:
        counts[np.argmin(np.abs(candidate_points - p))] += 1
    n_eff = min(n, candidate_points.size)
    top_n_points = candidate_points[np.argsort(counts)[::-1][:n_eff]]

    # round 2: consistent voters -> ranked by proximity
    sampled_points = sample_from(voter_dist, consistent_voters, 0.0, 1.0)
    ballot_counts = {}
    for p in sampled_points:
        order = tuple(top_n_points[np.argsort(np.abs(top_n_points - p))])
        ballot_counts[order] = ballot_counts.get(order, 0) + 1

    # inconsistent voters -> pick from "inconsistent" permutations bucketed by first choice
    perms = generate_and_group_permutations(top_n_points, n_eff)
    for key in list(perms.keys()):
        try:
            # if you have is_consistent(), filter; else keep as-is
            perms[key] = [b for b in perms[key] if not is_consistent(b, {t: t for t in top_n_points})]
        except NameError:
            pass

    inc = 1000 - consistent_voters
    if inc > 0:
        inc_pts = sample_from(voter_dist, inc, 0.0, 1.0)
        for p in inc_pts:
            anchor = top_n_points[np.argmin(np.abs(top_n_points - p))]
            bag = perms.get(anchor) or []
            if bag:
                ranking = random.choice(bag)
                ballot_counts[ranking] = ballot_counts.get(ranking, 0) + 1

    # run election (assumes you have voting_rules implemented)
    election = voting_rules(ballot_counts, top_n_points)
    irv_winner = election.irv()[0]
    condorcet_winner = election.condorcet()
    plurality_winner = election.plurality()
    first_place_support = election.get_first_place()
    if condorcet_winner == -1:
        first_place_support[-1] = -1

    if visualize:
        plt.figure(figsize=(12,4))
        plt.plot(x_can, y_can, label='Candidate distribution')
        plt.scatter(candidate_points, np.zeros_like(candidate_points), label='Random candidates', zorder=5)
        plt.scatter(top_n_points, np.zeros_like(top_n_points), marker='x', s=100, label='Top N', zorder=6)
        plt.legend(); plt.xlabel('X'); plt.ylabel('Density'); plt.title('Candidate distribution'); plt.tight_layout(); plt.show()

        plt.figure(figsize=(12,4))
        plt.plot(x_voter_plot, y_voter_plot, label='Voter KDE')
        plt.axvline(x=np.median(sampled_points_primary), linestyle='--', label='Median voter')
        plt.legend(); plt.xlabel('Value'); plt.ylabel('Density'); plt.title('Voter distribution'); plt.tight_layout(); plt.show()

    return (
        irv_winner, plurality_winner, condorcet_winner, top_n_points,
        random_gamma,
        first_place_support.get(condorcet_winner, -1),
        first_place_support.get(irv_winner, -1),
        first_place_support.get(plurality_winner, -1),
    )


def fd_bins(x):
    """Freedman–Diaconis bin count (clamped)."""
    x = np.asarray(x, float)
    if x.size < 2:
        return 10
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    bw = 2 * iqr * (x.size ** (-1/3))
    if bw <= 0:
        return 30
    b = int(np.ceil((x.max() - x.min()) / bw))
    return int(np.clip(b, 20, 120))

def hdi(vals, alpha=0.05):
    """Return (low, high) for the narrowest 1-alpha interval (HDI)."""
    v = np.sort(np.asarray(vals, float))
    n = v.size
    if n == 0:
        return (np.nan, np.nan)
    m = int(np.floor((1 - alpha) * n))
    if m < 1:
        return (v[0], v[-1])
    widths = v[m:] - v[:n - m]
    j = np.argmin(widths)
    return (v[j], v[j + m])