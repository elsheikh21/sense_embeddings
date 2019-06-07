import argparse
import logging
import multiprocessing
import ast
from glob import iglob

import numpy as np
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from scipy.stats import spearmanr
from tqdm import tqdm
from utils import initialize_logger, save_to_file

from directory_vars import directory_variables
from parsing_dataset import parse_datasets, remove_stopwords
from score import compute_scores, parse_tab_file


class Compute_Loss(CallbackAny2Vec):
    '''
    Callback to print loss after each epoch.
    '''

    def __init__(self):
        self.epoch = 1

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        print(f'Loss after epoch {self.epoch}: {loss}')
        self.epoch += 1


def build_model(sentences, window_size, sample):
    """Building the word2vec model, using multicores, as well as, building the
    model vocabulary.

    Arguments:
        sentences: {[list of lists]} -- containing all the corpus sentences
        with lemma_synset. if the synsets are present in the wordnet
        neither lemmatized, nor skolematized.
    Returns:
        w2v_model: Gensim word2vec model
    """
    logging.info('Building the model')
    cores = multiprocessing.cpu_count()
    w2v_model = Word2Vec(window=window_size, size=400, sample=sample,
                         workers=cores - 1, compute_loss=True, negative=5,
                         alpha=0.001, callbacks=[Compute_Loss()], hs=1)
    logging.info('Model is built')

    logging.info('Building model vocabulary')
    w2v_model.build_vocab(sentences, progress_per=1_000_000)
    logging.info('Model vocabulary is built')
    return w2v_model


def train_model(w2v_model, sentences, epochs, model_path):
    """Train the model passed, saves the output in form of sense embeddings
    only.

    Arguments:
        - w2v_model: gensim model -- to train
        - w2v_model: gensim model -- to train
        - sentences: list of lists -- containing all the corpus sentences
            with lemma_synset. if the synsets are present in the wordnet
            mapping provided, otherwise the word is passed as it is,
        - epochs: int
        - model_path: str -- stating where model to be saved, if none
            model is not saved.

    Returns:
        None
    """
    logging.info('Training model...')
    w2v_model.train(sentences,
                    total_examples=w2v_model.corpus_count,
                    epochs=epochs, compute_loss=True)
    logging.info('Done training model')
    if model_path:
        w2v_model.wv.save_word2vec_format(model_path, binary=False)
        sentences.clear()
        mask_embeddings(model_path)


def mask_embeddings(embeddings_path):
    """This method takes the saved model file and remove from it all the word
    embeddings leaving behind only the sense embeddings.

    Arguments:
        - embeddings_path: str

    Returns:
        - None
    """
    senses = []
    with open(embeddings_path, encoding='utf-8', mode='r') as embeddings_file:
        lines = embeddings_file.read().splitlines()
        for line in tqdm(lines, desc="Removing Word embeddings from file"):
            key = line.split(' ', 1)[0]
            if '_bn:' in key:
                senses.append(line)

    with open(embeddings_path, encoding='utf-8', mode='w') as file:
        file.write(f"{len(senses)} {len(senses[0].split(' ')) - 1}\n")
        for sense in tqdm(senses, desc='Writing sense embeddings file'):
            file.write(f'{sense}\n')


def eval_model(model, tab_file_path):
    """Evaluate the model, by parsing a tabular file, and fetching all the
    words computing for all pairs -as pair file- cosine similarity and
    computing the spearman correlation after that.

    Arguments:
        - tab_file_path: str -- path to tabular file

    Returns:
        - spearman_correlation: float
    """
    (word1_list, word2_list,
     gold_scores_arr) = parse_tab_file(tab_file_path)

    cosine_scores_arr = compute_scores(model,
                                       word1_list,
                                       word2_list,
                                       weighted=True)

    spearman_correlation = spearmanr(gold_scores_arr,
                                     cosine_scores_arr)

    return np.round(spearman_correlation[0], 4)


def train_multiple_corpus(path, w2v_model, tab_file_path, param, epochs=50):
    """
    loads multiple corpuses and retrain on them in case of a corpus output
    is too large to be loaded at once into memory, or in case of training
    on different corpora.
    After each training on file, it evaluates the model

    Arguments:
        - path: str -- path of the corpora
        - w2v_model: Gensim model object -- used to update the given model
        - tab_file_path: str -- tabular file path for evaluation
        - param: output of arguments parser object
        - epochs=50: int -- optional parameter

    Returns:
        - w2v_model: gensim model
    """
    for i, file in tqdm(enumerate(iglob(path)), desc="Updating Corpus"):
        lines = parse_file(file)
        w2v_model.build_vocab(lines, update=True)
        train_model(w2v_model, lines, epochs, model_path)
        lines.clear()
        spearman_correlation = eval_model(w2v_model, tab_file_path)
        print(
            f"Model eval after {i + 1} new sentences: {spearman_correlation}")
    return w2v_model


def parse_file(path):
    """It parses the corpus file into a list of lists,
    used for experimental purposes instead of parsing the
    whole corpus every time.

    Arguments:
        path {[str]} -- [path to corpus output]

    Returns:
        [list of lists] -- [sentences of the corpus]
    """
    sentences = []
    with open(path, encoding='utf-8', mode='r') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc=f'Fetching {path[path.rfind("/") + 1:]} output'):
            line_ = ast.literal_eval(line)
            sentences.append(line_)
    save_to_file(sentences, path)
    return sentences


def parse_arg():
    parser = argparse.ArgumentParser(description="Train Sense Embedding model")

    # Hyper-parameters
    parser.add_argument("--window", type=int, default=10)

    # Training vars
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--sample", type=float, default=1e-5)

    params = vars(parser.parse_args())
    return params


if __name__ == "__main__":
    np.random.seed(0)

    initialize_logger()

    params = parse_arg()

    (file_path, train_o_matic_file_path, sew_dir_path, output_file_path,
     mapping_file, filtered_corpus_output,
     model_path, tab_file_path) = directory_variables()

    sentences = parse_datasets(file_path, train_o_matic_file_path, sew_dir_path,
                               output_file_path, mapping_file)

    w2v_model = build_model(sentences, params["window"], params["sample"])

    w2v_model = train_multiple_corpus(
        sew_dir_path, w2v_model, tab_file_path, params)

    train_model(w2v_model, sentences, params["epochs"], model_path)

    spearman_correlation = eval_model(w2v_model, tab_file_path)

    print(f"Spearman corr is: {spearman_correlation}")
