#
# Copyright 2017 Benjamin Kiessling
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.

# -*- coding: utf-8 -*-
"""
Decoders for softmax outputs of CTC trained networks.
"""

import collections
import numpy as np

from typing import List, Tuple
from scipy.special import logsumexp
from scipy.ndimage import measurements

from itertools import groupby
from kraken.lib.lm import KrakenInterpolatedLM

__all__ = ['beam_decoder', 'greedy_decoder', 'blank_threshold_decoder']


def beam_decoder(outputs: np.ndarray, beam_size: int = 3) -> List[Tuple[int, int, int, float]]:
    """
    Translates back the network output to a label sequence using
    same-prefix-merge beam search decoding as described in [0].

    [0] Hannun, Awni Y., et al. "First-pass large vocabulary continuous speech
    recognition using bi-directional recurrent DNNs." arXiv preprint
    arXiv:1408.2873 (2014).

    Args:
        output (numpy.array): (C, W) shaped softmax output tensor

    Returns:
        A list with tuples (class, start, end, prob). max is the maximum value
        of the softmax layer in the region.
    """
    c, w = outputs.shape
    probs = np.log(outputs)
    beam = [(tuple(), (0.0, float('-inf')))]  # type: List[Tuple[Tuple, Tuple[float, float]]]

    # loop over each time step
    for t in range(w):
        next_beam = collections.defaultdict(lambda: 2*(float('-inf'),))  # type: dict
        # p_b -> prob for prefix ending in blank
        # p_nb -> prob for prefix not ending in blank
        for prefix, (p_b, p_nb) in beam:
            # only update ending-in-blank-prefix probability for blank
            n_p_b, n_p_nb = next_beam[prefix]
            n_p_b = logsumexp((n_p_b, p_b + probs[0, t], p_nb + probs[0, t]))
            next_beam[prefix] = (n_p_b, n_p_nb)
            # loop over non-blank classes
            for s in range(1, c):
                # only update the not-ending-in-blank-prefix probability for prefix+s
                l_end = prefix[-1][0] if prefix else None
                n_prefix = prefix + ((s, t, t),)
                n_p_b, n_p_nb = next_beam[n_prefix]
                if s == l_end:
                    # substitute the previous non-blank-ending-prefix
                    # probability for repeated labels
                    n_p_nb = logsumexp((n_p_nb, p_b + probs[s, t]))
                else:
                    n_p_nb = logsumexp((n_p_nb, p_b + probs[s, t], p_nb + probs[s, t]))

                next_beam[n_prefix] = (n_p_b, n_p_nb)

                # If s is repeated at the end we also update the unchanged
                # prefix. This is the merging case.
                if s == l_end:
                    n_p_b, n_p_nb = next_beam[prefix]
                    n_p_nb = logsumexp((n_p_nb, p_nb + probs[s, t]))
                    # rewrite both new and old prefix positions
                    next_beam[prefix[:-1] + ((prefix[-1][0], prefix[-1][1], t),)] = (n_p_b, n_p_nb)
                    next_beam[n_prefix[:-1] + ((n_prefix[-1][0], n_prefix[-1][1], t),)] = next_beam.pop(n_prefix)

        # Sort and trim the beam before moving on to the
        # next time-step.
        beam = sorted(next_beam.items(),
                      key=lambda x: logsumexp(x[1]),
                      reverse=True)
        beam = beam[:beam_size]
    return [(c, start, end, max(outputs[c, start:end+1])) for (c, start, end) in beam[0][0]]


def greedy_decoder(outputs: np.ndarray) -> List[Tuple[int, int, int, float]]:
    """
    Translates back the network output to a label sequence using greedy/best
    path decoding as described in [0].

    [0] Graves, Alex, et al. "Connectionist temporal classification: labelling
    unsegmented sequence data with recurrent neural networks." Proceedings of
    the 23rd international conference on Machine learning. ACM, 2006.

    Args:
        output (numpy.array): (C, W) shaped softmax output tensor

    Returns:
        A list with tuples (class, start, end, max). max is the maximum value
        of the softmax layer in the region.
    """
    labels = np.argmax(outputs, 0)
    seq_len = outputs.shape[1]
    mask = np.eye(outputs.shape[0], dtype='bool')[labels].T
    classes = []
    for label, group in groupby(zip(np.arange(seq_len), labels, outputs[mask]), key=lambda x: x[1]):
        lgroup = list(group)
        if label != 0:
            classes.append((label, lgroup[0][0], lgroup[-1][0], max(x[2] for x in lgroup)))
    return classes

from copy import copy

class Beam:

    def __init__(self, lm, alpha, classes=None, score=0, last_label=-1):
        self.score = score
        self.classes = [] if classes is None else classes
        self.last_label = last_label
        self.lm = lm
        self.alpha = alpha

    def get_extended_score(self, label, prob):
        indices = [idx for idx, _, _, _ in self.classes] + [label]
        ngram = self.lm.indices2ngram(indices)
        return self.score + prob + self.lm.log_p(ngram) * self.alpha

    def extend(self, label, prob):
        new_score = self.get_extended_score(label, prob)
        score_delta = new_score - self.score

        if label == 0:
            self.last_label = -1
        else:
            t = 0 if len(self.classes) == 0 else self.classes[-1][2] + 1
            if self.last_label == label:
                L, start, end, S = self.classes[-1]
                self.classes[-1] = (L, start, t, max(S, score_delta))
            else:
                class_tuple = (label, t, t, score_delta)
                self.classes.append(class_tuple)
            self.last_label = label
        
        self.score = new_score

        return Beam(self.lm, self.alpha, copy(self.classes), self.score, self.last_label)


def custom_decoder(outputs, codec, alpha=0.5, beam_size=1):
    lm = KrakenInterpolatedLM(codec)
    probs = np.log(outputs)
    n_vocab = outputs.shape[0]
    seq_len = outputs.shape[1]
    beams = [Beam(lm, alpha) for _ in range(beam_size)]
    for t in range(seq_len):
        empty_prob = probs[0, t]
        greedy_label = np.argmax(probs[:, t])
        if greedy_label == 0:
            for beam in beams:
                beam.extend(0, empty_prob)
        else:
            candidates = [(b, s) for b in range(beam_size) for s in range(1, n_vocab)]
            sorted_candidates = sorted(candidates, key=lambda x: beams[x[0]].get_extended_score(x[1], probs[x[1], t]))
            beams = [beams[b].extend(s, probs[s, t]) for b, s in sorted_candidates[-beam_size:]]
    return beams[-1].classes

def custom_decoder2(outputs, codec, beam_size=5, alpha=1):
    # adapted beam search, using LM

    lm = KrakenInterpolatedLM(codec)

    c, w = outputs.shape
    probs = np.log(outputs)
    beam = [(tuple(), (0.0, float('-inf')))]  # type: List[Tuple[Tuple, Tuple[float, float]]]

    # loop over each time step
    for t in range(w):
        next_beam = collections.defaultdict(lambda: 2*(float('-inf'),))  # type: dict
        # p_b -> prob for prefix ending in blank
        # p_nb -> prob for prefix not ending in blank
        for prefix, (p_b, p_nb) in beam:

            ngram = lm.prefix2ngram(prefix)

            # only update ending-in-blank-prefix probability for blank
            n_p_b, n_p_nb = next_beam[prefix]
            n_p_b = logsumexp((n_p_b, p_b + probs[0, t], p_nb + probs[0, t]))
            next_beam[prefix] = (n_p_b, n_p_nb)
            # loop over non-blank classes
            for s in range(1, c):
                # only update the not-ending-in-blank-prefix probability for prefix+s
                l_end = prefix[-1][0] if prefix else None
                n_prefix = prefix + ((s, t, t),)
                n_p_b, n_p_nb = next_beam[n_prefix]

                dp = probs[s, t] + alpha * (lm.log_p(ngram) - lm.log_p(ngram[-1:]))

                if s == l_end:
                    # substitute the previous non-blank-ending-prefix
                    # probability for repeated labels
                    n_p_nb = logsumexp((n_p_nb, p_b + dp))
                else:
                    n_p_nb = logsumexp((n_p_nb, p_b + dp, p_nb + dp))

                next_beam[n_prefix] = (n_p_b, n_p_nb)

                # If s is repeated at the end we also update the unchanged
                # prefix. This is the merging case.
                if s == l_end:
                    n_p_b, n_p_nb = next_beam[prefix]
                    n_p_nb = logsumexp((n_p_nb, p_nb + dp))
                    # rewrite both new and old prefix positions
                    next_beam[prefix[:-1] + ((prefix[-1][0], prefix[-1][1], t),)] = (n_p_b, n_p_nb)
                    next_beam[n_prefix[:-1] + ((n_prefix[-1][0], n_prefix[-1][1], t),)] = next_beam.pop(n_prefix)

        # Sort and trim the beam before moving on to the
        # next time-step.
        beam = sorted(next_beam.items(),
                      key=lambda x: logsumexp(x[1]),
                      reverse=True)
        beam = beam[:beam_size]
    return [(c, start, end, max(outputs[c, start:end+1])) for (c, start, end) in beam[0][0]]
    #return greedy_decoder(outputs)

def blank_threshold_decoder(outputs: np.ndarray, threshold: float = 0.5) -> List[Tuple[int, int, int, float]]:
    """
    Translates back the network output to a label sequence as the original
    ocropy/clstm.

    Thresholds on class 0, then assigns the maximum (non-zero) class to each
    region.

    Args:
        output (numpy.array): (C, W) shaped softmax output tensor
        threshold (float): Threshold for 0 class when determining possible
                           label locations.

    Returns:
        A list with tuples (class, start, end, max). max is the maximum value
        of the softmax layer in the region.
    """
    outputs = outputs.T
    labels, n = measurements.label(outputs[:, 0] < threshold)
    mask = np.tile(labels.reshape(-1, 1), (1, outputs.shape[1]))
    maxima = measurements.maximum_position(outputs, mask, np.arange(1, np.amax(mask)+1))
    p = 0
    start = None
    x = []
    for idx, val in enumerate(labels):
        if val != 0 and start is None:
            start = idx
            p += 1
        if val == 0 and start is not None:
            if maxima[p-1][1] == 0:
                start = None
            else:
                x.append((maxima[p-1][1], start, idx, outputs[maxima[p-1]]))
                start = None
    # append last non-zero region to list of no zero region occurs after it
    if start:
        x.append((maxima[p-1][1], start, len(outputs), outputs[maxima[p-1]]))
    return [y for y in x if x[0] != 0]
