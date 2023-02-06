from datetime import datetime

from functools import partial
from nltk.util import bigrams
from nltk.lm import MLE
from nltk.lm.preprocessing import pad_both_ends, flatten


def calculate_age(date1, string2):
    date2 = datetime.strptime(string2, '%a %b %d %H:%M:%S %z %Y')
    return (date1 - date2).days


def create_language_model(username_list):
    lm = MLE(2)

    padding_fn = partial(pad_both_ends, n=2)
    bigram_list = [list(bigrams(list(padding_fn(name.split("_"))))) for name in username_list]
    vocab = list(flatten(map(lambda s: padding_fn(s.split("_")), username_list)))

    lm.fit(bigram_list, vocab)
    return lm


def calculate_likelihood(lm, name):
    lk_scores = [lm.counts[[a]][b] for a, b in bigrams(list(pad_both_ends(name.split("_"), n=2)))]
    mean = sum(lk_scores) / len(lk_scores)

    return mean