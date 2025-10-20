import argparse
import seaborn as sns
from scipy.optimize import linprog
from scipy.stats import kurtosis, skew
import os 
from rcv_distribution import *
from MDS_analysis import *
from consistency import *
from voting_rules import *
from itertools import permutations
from collections import defaultdict
from collections import Counter
import random


# capturing lengths specific to each candidate
def get_mild_null_ballots(df):

    ballots = {}
    candidates = df['candidate'].values

    for c in candidates:
        ballots[(c,)] = df.loc[df['candidate']==c, 'len_1'].values[0]
        b = (c,)
    
        for i in range(2, len(candidates) + 1):
            if df.loc[df['candidate']==c, 'len_'+str(i)].values[0] != -1:
                count_i = df.loc[df['candidate']==c, 'len_'+str(i)].values[0]
                
                while(len(b) < i):
                    random_candidate = random.choice(candidates)
                    if random_candidate not in b:
                        b += (random_candidate,)
    
                ballots[b] = count_i
    return ballots


#cepturing the lengths
def get_slow_null_ballots(df):
    
    final_ballots = {}
    ballots_set = Counter()
    candidates = df['candidate'].values
    candidates = sorted(candidates)


    for candidate in candidates:
        first_place = df.loc[df['candidate']==candidate, 'first place count'].values[0]
        ballots_set[(candidate,)] += first_place


    lengths = {}
    for i in range (len(candidates)):
        sum_i = df['len_' + str(i+1)].sum()
        if (sum_i > 0):
            lengths[i+1] = sum_i

    for i in range(len(candidates)):
        if  i+1 == 1:
            ballots_list = list(ballots_set.elements())
            random_voters = random.choices(ballots_list, k=lengths[i+1])
            for b in random_voters:
                if b not in final_ballots:
                    final_ballots[b] = 0
                final_ballots[b] += 1
                ballots_set[b] -= 1
            

        elif i+1 in lengths and lengths[i+1] > 0:
            len_i = lengths[i+1]
            ballots_list = list(ballots_set.elements())
            random_voters = random.choices(ballots_list, k=lengths[i+1])

            for ranking in random_voters:
                new_ranking = ranking
                random_candidates = random.sample(candidates, i)
                for c in random_candidates:
                    new_ranking += (c,)
                if new_ranking not in final_ballots:
                    final_ballots[new_ranking] = 0
                final_ballots[new_ranking] += 1
                ballots_set[ranking] -= 1
           
   
    return final_ballots



def generate_and_group_permutations(items, choices, min_len=1):
    """
    Group all permutations of 'items' by their first element, for lengths in
    [min_len, choices]. Example group key: the first candidate in the ballot.
    """
    items = list(items)
    Lmin = max(1, int(min_len))
    Lmax = min(len(items), int(choices))
    grouped = defaultdict(list)

    if Lmin > Lmax:
        return grouped  # nothing to generate

    for L in range(Lmin, Lmax + 1):
        for perm in permutations(items, L):
            grouped[perm[0]].append(perm)
    return grouped


def get_extreme_null_ballots(df, choices, rng=None):
    """
    Build a dict of ballots -> counts.
    Uses only ballots with length > 2 (i.e., length >= 3).
    """
    if rng is None:
        rng = random  # use global RNG unless one is passed in

    candidates = sorted(df['candidate'].astype(str).values)

    # Generate only length >= 3 ballots, grouped by first choice
    grouped_perms = generate_and_group_permutations(candidates, choices, min_len=3)

    ballots = {}
    # If you actually want to use 'first place count' instead of freq, switch below.
    for cand in candidates:
        # how many first-place ballots to draw for this candidate
        first_place_freq = float(df.loc[df['candidate'] == cand, 'first place freq'].values[0])
        first_place = int(round(first_place_freq, 3) * 1000)

        # If there are no length>=3 permutations starting with this candidate, skip
        pool = grouped_perms.get(cand, [])
        if not pool or first_place <= 0:
            continue

        # sample permutations with replacement
        sampled = rng.choices(pool, k=first_place)
        for b in sampled:
            ballots[b] = ballots.get(b, 0) + 1

    return ballots

def get_null_gamma(filename, randomness):

    directory = "null_elections"
    csv = os.path.join(directory, filename) 
    df = pd.read_csv(csv)  
    candidates = df['candidate'].values
    election = pd.read_csv("election_table.csv")
    choices = min(len(candidates), round(election.loc[election['filename']==filename, 'choices'].values[0]))
    
    #print(choices)
    if randomness == 'extreme':
        ballots = get_extreme_null_ballots(df, choices)
    elif randomness == 'slow':
        ballots = get_slow_null_ballots(df)
    else:
        ballots = get_mild_null_ballots(df)

    normalized_distances_original = {}
    for c in candidates:
        position = df.loc[df['candidate']==c, 'position']
        normalized_distances_original[c] = position.values[0]
    #print(filename, "  original normalized distances: ", normalized_distances_original)

    test = perform_rcv_analysis(ballots, candidates, n_init=100, max_itr=1000, n_runs=1000, metric=False)
    mds_1d_coordinates, mds_2d_coordinates, most_common_order, order_frequencies, candidate_names = test

    # Print the normalized distances between candidates and plot the MDS analysis
    normalized_distances = get_distances_normalized(most_common_order, mds_1d_coordinates, candidate_names)

    #print("new distances: ", normalized_distances_original)
    consistent_ballots, gamma = get_permissive_gamma(ballots, normalized_distances)
    return consistent_ballots, gamma


def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('input', metavar='INPUT', type=str, help='Input string')
    
    args = parser.parse_args()
    user_input = args.input

    null_gammas = pd.read_csv("null_gammas.csv")
    ans = []
    sum = 0
    for i in range(1, 1001):
        sum += get_null_gamma(user_input)[1]
        if i == 10:
            ans.append(sum/10)
            null_gammas.loc[null_gammas["filename"] == user_input, "null_gamma_10"] = sum/10
        if i == 100:
            ans.append(sum/100)
            null_gammas.loc[null_gammas["filename"] == user_input, "null_gamma_100"] = sum/100
    ans.append(sum/1000)
    null_gammas.loc[null_gammas["filename"] == user_input, "null_gamma_1000"] = sum/1000
    
    
    null_gammas.to_csv("null_gammas.csv", index=False)  # Ensure index is not written to the CSV


    
    

if __name__ == "__main__":
    main()
