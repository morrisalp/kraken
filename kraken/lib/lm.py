# language model

import pickle
import pathlib
import os
import json
import numpy as np

curdir = pathlib.Path(__file__).parent.absolute()

class KrakenInterpolatedLM:

    UNK_CHAR = '^'
    PAD_CHAR = '`'

    def __init__(self):

        
        with open(os.path.join(curdir, 'coeff_search.json'), 'r') as f:
            coeff_search = json.load(f)
        Ls = coeff_search['best_Ls']

        with open(os.path.join(curdir, 'ngram_totals.pk'), 'rb') as f:
            ngram_totals = pickle.load(f)

        with open(os.path.join(curdir, 'ngram_counts.pk'), 'rb') as f:
            ngram_counts = pickle.load(f)

        ns_calculated = set(ngram_totals.keys())

        assert all(n in ns_calculated for n in range(1, len(Ls) + 1)), (
            'Some n values are uncalculated in NgramCalculator'
        )
        assert np.allclose(sum(Ls), 1), f'Ls do not add up to 1; sum={sum(Ls)}'
        self.Ls = Ls # linear interpolation weights
        self.N = len(Ls)


    def calculate_p(self, ngram, epsilon=1):
        n = len(ngram)
        assert n in self.ns_calculated, f"Missing calculation for n={len(ngram)}"
        ngram_pref = ngram[:-1]
        num = self.ngram_counts[n].get(ngram, 0)
        
        if n == 1:
            denom = self.ngram_totals[n]
        else:
            assert n - 1 in self.ns_calculated, f"Missing calculation for n={n - 1}"
            denom = self.ngram_counts[n-1].get(ngram_pref, 0)
        
        epsilon_normed = epsilon / N_CHARS ** n
        return (num + epsilon_normed) / (denom + epsilon_normed * N_CHARS ** n) # additive smoothing

    def log_p(self, ngram_string):
        ngram = tuple(ngram_string)
        # allow shorter ngrams, only using some interpolation coefficients
        n = len(ngram)
        assert n <= self.N, f'Wrong ngram length: {n} > {self.N}'
        ngrams = [ngram[-i-1:] for i in range(n)]
        probs = [self.calculate_p(ng) for ng in ngrams]
        weights = np.array(self.Ls[:n]) / sum(self.Ls[:n])
        weighted_probs = [p * L for p, L in zip(probs, weights)]
        avg_prob = sum(weighted_probs)
        return np.log2(avg_prob)


lm = KrakenInterpolatedLM()
