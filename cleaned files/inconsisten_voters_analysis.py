def kendall_tau_distance(order1, order2):
    """
    Compute Kendall–Tau distance (number of pairwise disagreements)
    between two rankings over the same set of items.

    order1, order2: sequences of the same items, no duplicates.
    """
    if len(order1) != len(order2):
        raise ValueError("kendall_tau_distance expects same-length permutations")

    pos2 = {c: i for i, c in enumerate(order2)}
    inv = 0
    m = len(order1)
    for i in range(m):
        ci = order1[i]
        for j in range(i + 1, m):
            cj = order1[j]
            # disagreement if order1 says ci before cj but order2 says cj before ci
            if pos2[ci] > pos2[cj]:
                inv += 1
    return inv


def compute_consistent_full_orders(normalized_distances):
    """
    Given normalized_distances: dict[candidate] -> position on 1D axis,
    compute all distinct *full* rankings of all candidates that can arise
    by sorting candidates by distance to some ideal point x on the line.

    Returns:
        list of tuples, each tuple is a permutation of candidates,
        representing a distance-based consistent full order.
    """
    pos = normalized_distances
    candidates = list(pos.keys())
    n = len(candidates)

    if n <= 1:
        # Only one candidate: only one order
        return [tuple(candidates)]

    # Axis order for deterministic tie-breaking
    axis_order = sorted(candidates, key=lambda c: pos[c])
    axis_index = {c: i for i, c in enumerate(axis_order)}

    # All pairwise midpoints
    positions = [pos[c] for c in candidates]
    midpoints = sorted({
        (positions[i] + positions[j]) / 2.0
        for i in range(n)
        for j in range(i + 1, n)
    })

    # Build intervals (-∞, mid1), (mid1, mid2), ..., (mid_k, +∞)
    # We'll take one sample x in each interval
    min_pos, max_pos = min(positions), max(positions)
    breaks = [min_pos - 1.0] + midpoints + [max_pos + 1.0]
    sample_xs = [(breaks[k] + breaks[k + 1]) / 2.0 for k in range(len(breaks) - 1)]

    def full_order_for_x(x):
        # sort by distance to x, breaking ties with axis order
        return tuple(sorted(
            candidates,
            key=lambda c: (abs(pos[c] - x), axis_index[c])
        ))

    seen = set()
    orders = []
    for x in sample_xs:
        order = full_order_for_x(x)
        if order not in seen:
            seen.add(order)
            orders.append(order)

    return orders


def off_dim_distance_for_ballot(ballot, consistent_full_orders):
    """
    Compute the (raw and normalized) Kendall–Tau distance from a ballot
    to its closest 'consistent' ballot, where consistency means:

        the ranking is induced by distance to some ideal point x
        on the 1D axis used to build `consistent_full_orders`.

    Parameters
    ----------
    ballot : iterable of candidates
        e.g. ('Alice','Bob','Carol') or ['A','C','D'].
    consistent_full_orders : list of tuples
        Output of compute_consistent_full_orders(normalized_distances).

    Returns
    -------
    raw_distance : int
        Minimum Kendall–Tau distance to any consistent ranking
        (restricted to this ballot's candidates).
    normalized_distance : float
        raw_distance divided by C(m, 2), where m = len(ballot). In [0,1].
        For len(ballot) <= 2, returns (0, 0.0).
    """
    b = list(ballot)
    m = len(b)

    # With ≤2 candidates, there's no meaningful "shape" to violate;
    # we treat them as perfectly on-dimension.
    if m <= 2:
        return 0, 0.0

    best_raw = None

    for full_order in consistent_full_orders:
        # project full order down to this ballot's candidate set
        proj = [c for c in full_order if c in b]

        # sanity check: proj should contain exactly the ballot's items (maybe in different order)
        if len(proj) != m:
            # This can happen if ballot contains candidates not in the axis dict; skip those
            continue

        d = kendall_tau_distance(b, proj)
        if best_raw is None or d < best_raw:
            best_raw = d
            if best_raw == 0:
                break  # can't do better

    # If for some reason we never matched (e.g., ballot contains unknown candidates)
    if best_raw is None:
        return 0, 0.0

    max_pairs = m * (m - 1) // 2
    if max_pairs == 0:
        norm = 0.0
    else:
        norm = best_raw / max_pairs

    return best_raw, norm
def compute_election_offdim_stats(ballots, normalized_distances):
    """
    For one election:

    - build all consistent full orders from `normalized_distances`
    - for each ballot, compute distance to nearest consistent order
    - treat ballots with distance 0 as consistent ("on-dimension")
    - compute gamma and average normalized distance among inconsistent ballots.

    Parameters
    ----------
    ballots : dict
        Mapping ballot -> count of voters with that ranking.
        e.g. {('A','B','C'): 10, ('B','A','C'): 5, ...}
    normalized_distances : dict
        Mapping candidate -> position on the axis.

    Returns
    -------
    stats : dict with keys
        - 'gamma' : fraction of voters with distance 0 (consistent)
        - 'avg_offdim_norm' : average normalized distance among inconsistent ballots
        - 'inconsistent_votes' : number of voters with non-zero distance
        - 'total_votes' : total number of voters
        - 'consistent_full_orders' : the list of distance-based full orders
    """
    # Precompute the consistent orders using the axis
    consistent_full_orders = compute_consistent_full_orders(normalized_distances)

    total_votes = 0
    inconsistent_votes = 0
    weighted_norm_sum = 0.0

    for ballot, count in ballots.items():
        m = len(ballot)
        if m == 0:
            continue  # skip empty ballots
        total_votes += count

        raw, norm = off_dim_distance_for_ballot(ballot, consistent_full_orders)

        if raw == 0:
            # treat as consistent
            continue

        inconsistent_votes += count
        weighted_norm_sum += count * norm

    if total_votes > 0:
        gamma = (total_votes - inconsistent_votes) / total_votes
    else:
        gamma = 0.0

    if inconsistent_votes > 0:
        avg_offdim_norm = weighted_norm_sum / inconsistent_votes
    else:
        avg_offdim_norm = 0.0  # or float('nan'), depending on what you prefer

    return {
        "gamma": gamma,
        "avg_offdim_norm": avg_offdim_norm,
        "inconsistent_votes": inconsistent_votes,
        "total_votes": total_votes,
        "consistent_full_orders": consistent_full_orders,
    }
