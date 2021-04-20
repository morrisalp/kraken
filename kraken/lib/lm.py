# language model

import pickle
import pathlib
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

curdir = pathlib.Path(__file__).parent.absolute()

class KrakenInterpolatedLM:

    UNK_CHAR = '^'
    PAD_CHAR = '`'
    KNOWN_CHARS = set('₈₅+ á Tú⸢d₁o6à!>8Qm-1s*èš3:9<ṢKn.Bp(À54)rUìÉGktŠÌùṭI"lEMéx?RzN7e₄ṣiubḫʾg]ZÚ/Ía$SÁ[vDP2—È⸣qAL₆ḪíÙ0')
    N_CHARS = len(KNOWN_CHARS) + 2 # extra 2 for UNK & PAD

    def __init__(self, codec):

        self.codec = codec
        
        with open(os.path.join(curdir, 'coeff_search.json'), 'r') as f:
            coeff_search = json.load(f)
        self.Ls = coeff_search['best_Ls']#[:4] # linear interpolation weights
        #self.Ls = [x / sum(self.Ls) for x in self.Ls]

        with open(os.path.join(curdir, 'ngram_totals.pk'), 'rb') as f:
            self.ngram_totals = pickle.load(f)

        with open(os.path.join(curdir, 'ngram_counts.pk'), 'rb') as f:
            self.ngram_counts = pickle.load(f)

        self.ns_calculated = set(self.ngram_totals.keys())

        assert all(n in self.ns_calculated for n in range(1, len(self.Ls) + 1)), (
            'Some n values are uncalculated in NgramCalculator'
        )
        assert np.allclose(sum(self.Ls), 1), f'Ls do not add up to 1; sum={sum(self.Ls)}'
        self.N = len(self.Ls)

    def idx2char(self, idx):
        codec_char = self.codec.l2c.get(chr(idx))
        return codec_char if codec_char in self.KNOWN_CHARS else self.UNK_CHAR

    def indices2string(self, indices):
        return ''.join([self.idx2char(idx) for idx in indices])

    def indices2ngram(self, indices):
        txt = self.PAD_CHAR + self.indices2string(indices)
        return txt[-self.N:]

    def prefix2ngram(self, prefix):
        # prefix in format as used in ctc_decoder.py
        prefix_indices = [idx for idx, _, _ in prefix]
        return self.indices2ngram(prefix_indices)

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
        
        epsilon_normed = epsilon / self.N_CHARS ** n
        return (num + epsilon_normed) / (denom + epsilon_normed * self.N_CHARS ** n) # additive smoothing

    def p(self, ngram_string):
        ngram = tuple(ngram_string)
        # allow shorter ngrams, only using some interpolation coefficients
        n = len(ngram)
        assert n <= self.N, f'Wrong ngram length: {n} > {self.N}'
        ngrams = [ngram[-i-1:] for i in range(n)]
        probs = [self.calculate_p(ng) for ng in ngrams]
        weights = np.array(self.Ls[:n]) / sum(self.Ls[:n])
        weighted_probs = [p * L for p, L in zip(probs, weights)]
        avg_prob = sum(weighted_probs)
        return avg_prob

    def log_p(self, ngram_string):
        return np.log(self.p(ngram_string))


class KrakenInterpolatedRNN:
    UNK_CHAR = '^'
    PAD_CHAR = '`'
    KNOWN_CHARS = set(
        '₈₅+ á Tú⸢d₁o6à!>8Qm-1s*èš3:9<ṢKn.Bp(À54)rUìÉGktŠÌùṭI"lEMéx?RzN7e₄ṣiubḫʾg]ZÚ/Ía$SÁ[vDP2—È⸣qAL₆ḪíÙ0')
    N_CHARS = len(KNOWN_CHARS) + 2  # extra 2 for UNK & PAD
    MAXLEN = 100

    def __init__(self, codec):

        self.codec = codec

        with open(os.path.join(curdir, 'tokenizer.pk'), 'rb') as f:
            self.tokenizer = pickle.load(f)

        self.model = tf.keras.models.load_model(os.path.join(curdir, 'saved_model'))

        self.N = len(self.Ls)

    def idx2char(self, idx):
        codec_char = self.codec.l2c.get(chr(idx))
        return codec_char if codec_char in self.KNOWN_CHARS else self.UNK_CHAR

    def indices2string(self, indices):
        return ''.join([self.idx2char(idx) for idx in indices])

    def indices2ngram(self, indices):
        txt = self.PAD_CHAR + self.indices2string(indices)
        return txt[-self.N:]

    def prefix2ngram(self, prefix):
        # prefix in format as used in ctc_decoder.py
        prefix_indices = [idx for idx, _, _ in prefix]
        return self.indices2ngram(prefix_indices)

    def p(self, t):
        return self.model.predict(t)

    def log_p(self, text):

        next_char = text[-1]
        text = text[:-1]
        seqs = self.tokenizer.texts_to_sequences([PAD_CHAR + text])
        next_char = self.tokenizer.texts_to_sequences([next_char])

        t = pad_sequences(seqs, maxlen=MAXLEN, padding='post', truncating='post')
        probs = self.p(t)
        return np.log(softmax(probs))[0, len(text), next_char]
