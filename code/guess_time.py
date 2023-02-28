#!/usr/bin/python3
"""
Make a comparison between the expected number of enumeration guesses needed in
MATZOV and the actual number. MATZOV expects 2^{k_enum * H(chi_s)} number of
guesses, given k_enum positions to try for a secret key where coefficients are
distributed by chi_s.
"""
import argparse
from proba_utils import entropy, iter_law_convolution, lg


def convolute_implicit(dist_a, dist_b):
    """
    Convolutes two distributions which are suppressed by similar probability,
    i.e. the outcome is unknown but we maintain a list of probabilities
    together with the number of events with this probability. This is for
    dist_a, dist_b as well as the result.
    :param dist_a: distribution dist_a (in implicit form)
    :param dist_b: distribution dist_b (in implicit form)
    :returns: dist_a convoluted with dist_b (in implicit form)
    """
    dist_c = dict([])
    for prob_a in dist_a:
        for prob_b in dist_b:
            prob_c = prob_a * prob_b
            dist_c[prob_c] = dist_c.get(prob_c, 0) + dist_a[prob_a] * dist_b[prob_b]
    return dist_c


def average_guessing_time(dist):
    """
    Return the least number of candidates one has to test on average to find
    the chosen candidate, assuming the candidates are distributed by dist. The
    optimal strategy for this is to sort the candidates by decreasing
    probability of being chosen.
    :param dist: the distribution (in implicit form)
    :returns: expected number of candidates to test (>= 1) before the chosen one is found.
    """
    probs = list(dist)
    # Sort the probabilities descendingly
    probs.sort(reverse=True)

    mean, count = 0, 0
    for prob in probs:
        # If this candidate occurs once, add prob * (count+1)
        # If this candidate occurs twice, add prob * (count+1) + prob * (count+2)
        # ...
        # If it occurs dist[prob] many times, add
        #     prob * ((count+1) + (count+2) + ... + (count+dist[prob])
        #     =
        mean += prob * dist[prob] * (count + (dist[prob] + 1)/2)
        count += dist[prob]
    return mean


parser = argparse.ArgumentParser()
parser.add_argument('--eta', type=int, default=3, help='\\eta as used in Kyber')
ETA = parser.parse_args().eta

coeff_dist = iter_law_convolution({-1:.25, 0:.5, 1:.25}, ETA)
entropy_coeff = entropy(coeff_dist)

# Recording the detailed distribution B3^17 would already be unreasonable.
# Instead, we switch to a compact representation, counting how many values (C)
# have a specific probability p, as a dictionary {p:C}.
# This allows implicit computation of cartesian product, as well as computing
# guessing time.

dist_impl = dict([])
for (x, p) in coeff_dist.items():
    dist_impl[p] = 1 + dist_impl.get(p, 0)
# Start with a singleton distribution
X = {1: 1}

print(f"  k, k*H(B{ETA}), E[B{ETA}^k],   E - H")
for k in range(1, 41):
    X = convolute_implicit(X, dist_impl)
    expected = k * entropy_coeff
    actual = lg(average_guessing_time(X))

    PROB_SUM = 0.0
    for (p, C) in X.items():
        PROB_SUM += p*C
    assert abs(1 - PROB_SUM) < 1e-8
    print(f"{k:3}, {expected:7.3f}, {actual:7.3f}, {actual-expected:7.3f}")
print(f"Final sum of probabilities is given by {PROB_SUM}")
