import logging
import pandas as pd
import csv

import numpy as np
from gensim.models import KeyedVectors
from scipy.stats import spearmanr
from tqdm import tqdm
from utils import initialize_logger

from directory_vars import directory_variables


def measure_similarity(model, word1_senses, word2_senses, weighted):
    """Measure similarity between 2 word senses, if either of them is empty
    so it is -1.0, otherwise either it measures weighted cosine similarity or
    just the cosine similarity

    Arguments:
        model {Gensim model}
        word1_senses {list}
        word2_senses {list}
        weighted {bool}

    Returns:
        [list] -- [list of floats]
    """
    similarities = []
    score = -1.
    if weighted:
        for s1 in word1_senses:
            for s2 in word2_senses:
                score = max(score,
                            weighted_cosine_similarity(model, s1, word1_senses,
                                                       s2, word2_senses))
        similarities.append(score)
    else:
        for s1 in word1_senses:
            for s2 in word2_senses:
                score = max(score, model.wv.similarity(s1, s2))
        similarities.append(score)
    return similarities


def compute_scores(model, word1_list, word2_list, weighted=False):
    """Takes the model path and words list to compute similarity between them,
    by getting for every word from both lists all its equivalent sense
    embeddings (S1, S2) and for every (s1, s2) computes the similarity using
    'model.similarity()' and appends the maximum of all senses to a list.
    returns this list.

    Arguments:
        model {gensim.models.word2vec}
        word1_list {list}
        word2_list {list}

    Returns:
        [list] -- [scores between all the pair of words in both lists]
    """
    cosine_scores_arr = []
    for word1, word2 in tqdm(zip(word1_list, word2_list),
                             desc="Computing Similarities"):
        S1 = get_sense_embeddings(model, word1)
        S2 = get_sense_embeddings(model, word2)
        cosine_scores_arr.append(measure_similarity(model, S1, S2, weighted))
    return cosine_scores_arr


def parse_tab_file(tab_file_path):
    """Parses tabular file to create 3 lists, 2 of which contain words to check
    model performance, and the third contain gold scores.

    Arguments:
        - tab_file_path: str
        - word1_list: list -- contain all word1
        - word2_list: list -- contain all word2
        - values_list: list -- contain all gold scores as per word1 & word2
    """
    word1_list = []
    word2_list = []
    values_list = []
    values_list = []
    with open(tab_file_path, 'r') as tab_file:
        reader = csv.reader(tab_file, delimiter='\t')
        next(reader)
        for row in tqdm(reader, desc="Parsing tabular file"):
            word1_list.append(row[0].lower())
            word2_list.append(row[1].lower())
            values_list.append(row[2])
    return word1_list, word2_list, values_list


def get_embeddings_dict(path):
    """Loads the embeddings dictionary from a given path.

    Arguments:
        - path: str

    Returns:
        - embedding_dict: dict
    """
    embeddings_dict = dict()
    with open(path, encoding='utf-8', mode='r') as file:
        lines = file.read().splitlines()
        for line in tqdm(lines, desc="Creating embeddings dictionary"):
            key_val = line.split(" ", 1)
            key, val = key_val[0], np.fromstring(key_val[1], sep=' ')
            embeddings_dict[key] = val
    return embeddings_dict


def get_sense_embeddings(w2v_model, word):
    """Given the model, it fetches all similar words in the model vocabulary.

    Arguments:
        w2v_model {gensim_model}
        word {str}

    Returns:
        similar_words {list} -- [all similar senses to word]
    """
    vocab_list = w2v_model.wv.vocab
    similar_words = []
    for item in vocab_list:
        item_ = item.split('_bn:')[0].replace('_', ' ')
        if word == item_:
            similar_words.append(item)
    return similar_words


def sense_dominance(model, word_sense, word_senses):
    """
    computes the dominance of its sense s belongs to S, by dividing
    the frequency of s by the overall frequencies of all senses
    associated with the word

    Arguments:
        - model: gensim word2vec model
        - word_sense: str -- sense as per word
        - word_senses: list -- all senses as per word

    Returns:
        - float
    """
    sense_freq = model.wv.vocab[word_sense].count
    senses_freq = sum([model.wv.vocab[_word_sense].count
                       for _word_sense in word_senses])
    dominance = sense_freq / senses_freq
    return dominance


def weighted_cosine_similarity(model, s1, word1_senses, s2, word2_senses):
    """
    Computing the word similarities based on their senses, but
    taking into consideration their relative importance

    Arguments:
        - model: gensim word2vec model
        - s1: str -- sense as per word 1
        - word1_senses: list -- all senses as per word 1
        - s2: str -- sense as per word 2
        - word2_senses: list -- all senses as per word 2


    Returns:
        - float
    """
    w1_sense_dominance = sense_dominance(model, s1, word1_senses)
    w2_sense_dominance = sense_dominance(model, s2, word2_senses)
    alpha = 8
    closest_similarity = (model.similarity(s1, s2) ** alpha)
    return w1_sense_dominance * w2_sense_dominance * closest_similarity


def tanimoto_measure(word1, word2):
    """
    Method compares vector of words, using generalized version of
    Jaccard similarity.

    Arguments:
        - word1: numpy.ndarray
        - word2: numpy.ndarray

    Returns:
        - float -- tanimoto coefficient
    """
    numerator = np.dot(word1, word2)
    denomenator = ((np.linalg.norm(word1)) * 2 +
                   (np.linalg.norm(word2))*2 - numerator)
    return np.round(numerator / denomenator, 4)


def tanimoto_words_similarity(tanimoto_words_ex, tanimoto_val, show=True):
    """
    Creates a pandas dataframe to be plotted with how words are similar,
    auxiliary function used to demonstrate that same word but
    different contextual meanings are very different

    """
    for idx, elem in enumerate(tanimoto_words_ex):
        elem = elem[0]
        if idx == 0:
            df = pd.DataFrame(tanimoto_val.get(elem), columns=[
                ' ' + tanimoto_words_ex.get(elem), 'Tanimoto Measure'])
        else:
            df = pd.concat([df, pd.DataFrame(tanimoto_val.get(elem), columns=[
                ' ' + tanimoto_words_ex.get(elem), 'Tanimoto Measure'])],
                axis=1, join='inner')
    if show:
        print(df.head())


if __name__ == "__main__":
    initialize_logger()

    (_, _, _, _, _, _, model_path, tab_file_path) = directory_variables()

    word1_list, word2_list, gold_scores_arr = parse_tab_file(tab_file_path)

    model = KeyedVectors.load_word2vec_format(model_path, binary=False)

    word1 = model.wv.word_vec('bank_bn:00008363n', use_norm=False)
    word2 = model.wv.word_vec('interest_rate_bn:00047085n', use_norm=False)

    print(f'Tanimoto Measure: is {tanimoto_measure(word1, word2)}')

    tanimoto_words = {'bank_bn:00008363n': 'Bank as riverbank(bank_bn:00008363n)',
                      'bank_bn:00008364n': 'Bank as money Bank (bank_bn:00008364n)'}

    tanimoto_values = dict()
    word = 'bank_bn:00008364n'
    similar_words_lst = model.similar_by_word(word, topn=5)
    tanimoto_words_lst = list(tanimoto_words.keys())
    w1 = model.wv.word_vec(word, use_norm=False)
    for word_ in similar_words_lst:
        w2 = model.wv.word_vec(word_[0], use_norm=False)
        tanimoto_values[word_] = tanimoto_measure(w1, w2)

    tanimoto_words_similarity(similar_words_lst, tanimoto_values)

    cosine_scores_arr = compute_scores(
        model, word1_list, word2_list, weighted=True)

    spearman_correlation = spearmanr(gold_scores_arr, cosine_scores_arr)

    print(f"The corr is: {spearman_correlation[0]}")
