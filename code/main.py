import argparse

from utils import initialize_logger
from directory_Vars import directory_variables
from parsing_dataset import parse_datasets
from model import build_model, train_model, eval_model
from plot import fetch_clusters_2d, tsne_plot_similar_words
from grid_search_model import grid_search


def parse_arg():
    parser = argparse.ArgumentParser(description="Train Sense Embedding model")

    # Hyper-parameters
    parser.add_argument("--window", type=int, default=10)

    # Training vars
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--sample", type=float, default=1e-4)
    parser.add_argument("--grid_search", type=bool, default=False)

    params = vars(parser.parse_args())
    return params


if __name__ == "__main__":
    initialize_logger()

    params = parse_arg()

    (file_path, train_o_matic_file_path, sew_dir_path, output_file_path,
     mapping_file, filtered_corpus_output,
     model_path, tab_file_path) = directory_variables()

    sentences = parse_datasets(
        file_path, train_o_matic_file_path, sew_dir_path,
        output_file_path, mapping_file)

    if params["grid_search"]:
        hyper_params = {'window': [5, 10],
                        'alpha': [1e-3, 1e-5, 1e-6],
                        'sample': [1e-4, 1e-5, 1e-6],
                        'negative': list(range(1, 10)),
                        }
        grid_output = 'grid_search.json'
        best_grid, _ = grid_search(tab_file_path, hyper_params, sentences, grid_output)
        logging.info(f"Best params are: {best_grid}")
    else:
        w2v_model = build_model(sentences, params["window"], params["sample"])

        train_model(w2v_model, sentences, params["epochs"], model_path)

        spearman_correlation = eval_model(w2v_model, tab_file_path)

        print(f"Spearman corr is: {spearman_correlation}")

        keys = w2v_model.wv.vocab.keys()[:5]
        embeddings_en_2d, word_clusters = fetch_clusters_2d(
            w2v_model, keys, 10)
        tsne_plot_similar_words('Similar Sense Embeddings',
                                keys,
                                embeddings_en_2d,
                                word_clusters, 0.7, 'similar_words.png')
