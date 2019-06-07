import logging
import json

from gensim.models import Word2Vec
from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import multiprocessing

from utils import initialize_logger
from model import eval_model, mask_embeddings, parse_file


class Word2vecExtended(BaseEstimator):
    """
    Extended word2vec model for hyperparameters fine tuning
    """
    cores = multiprocessing.cpu_count()

    def __init__(self, window=5, epochs=50, min_count=5, size=400, alpha=0.025,
                 sample=0.001, negative=5, workers=cores):
        self.model = None
        self.window = window
        self.min_count = min_count
        self.size = size
        self.alpha = alpha
        self.sample = sample
        self.epochs = epochs
        self.negative = negative
        self.workers = workers

    def fit(self, sentences):
        # build model and its vocab
        self.model = Word2Vec(size=self.size, window=self.window,
                              alpha=self.alpha, min_count=self.min_count,
                              workers=self.workers, sample=self.sample,
                              negative=self.negative, iter=self.epochs)
        self.model.build_vocab(sentences, update=False, progress_per=1_000_000)
        # train
        self.model.train(sentences=sentences,
                         total_examples=self.model.corpus_count,
                         epochs=self.epochs)
        return self

    def eval(self, tab_file_path):
        return eval_model(self.model, tab_file_path)


def grid_search(tab_file_path, hyper_params, sentences, file_name=None):
    best_score, best_params = -1., None
    grids = ParameterGrid(hyper_params)
    logging.info(f"Number of combinations: {len(list(grids))}.\n")

    grids_list = []
    # loop on all the hyper parameters grids we have
    for grid in tqdm(grids, desc="Tuning hyperparameters"):
        # Initiate and train the model
        model = Word2vecExtended(**grid)
        model.fit(sentences)
        score = model.eval(tab_file_path)
        # Deleting model, to avoid conflictions
        del model
        grid.update({"correlation": score})
        grids_list.append(grid)
        # Keeping the best
        if score > best_score:
            logging.info(f"Model correlation: {score}")
            best_params = grid

        if file_name:
            with open(file_name, mode='w+') as handle:
                json.dump(grids, handle)

    return best_params, grids_list


if __name__ == "__main__":
    initialize_logger()
    tab_file_path = r'C:/Users/Sheikh/Documents/GitLab/word_sense_embeddings/resources/wordsim353/combined.tab'
    file_path = r'C:/Users/Sheikh/Documents/GitLab/word_sense_embeddings/resources/EuroSense/corpus_output.txt'
    sentences = parse_file(file_path)
    hyper_params = {'window': [5, 10],
                    'alpha': [1e-3, 1e-5, 1e-6],
                    'sample': [1e-4, 1e-5, 1e-6],
                    'negative': list(range(5, 11)),
                    }

    grid_output = 'grid_search.json'
    best_grid, _ = grid_search(
        tab_file_path, hyper_params, sentences, grid_output)
