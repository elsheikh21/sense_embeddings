import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from gensim.models import KeyedVectors
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from utils import initialize_logger
from sklearn.decomposition import PCA
import json
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from directory_vars import directory_variables


def fetch_clusters_2d(model, keys, top_n):
    """
    Fetch from model top_n similar words to the ones in keys list
    passed,

    Arguments:
        - model: gensim word2vec model
        - keys: list -- keys of word to fetch similar words to
        - top_n: int -- top n similar words for every word in keys

    Returns:
        - embeddings_en_2d: ndarray
        - word_clusters: ndarray
    """
    embedding_clusters = []
    word_clusters = []
    for word in keys:
        embeddings = []
        words = []
        for similar_word, _ in model.most_similar(word, topn=top_n):
            words.append(similar_word)
            embeddings.append(model[similar_word])
        embedding_clusters.append(embeddings)
        word_clusters.append(words)

    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    tsne_model_en_2d = TSNE(perplexity=15,
                            n_components=2,
                            init='pca',
                            n_iter=3500,
                            random_state=32,
                            verbose=1)
    embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(
        embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
    return embeddings_en_2d, word_clusters


def tsne_plot_similar_words(title, labels, embedding_clusters,
                            word_clusters, a, filename=None):
    """
    Visualizing high-dimensional Word2Vec word embeddings using t-SNE

    Arguments:
        - title: str
        - labels: list -- keys to plot similar words to
        - embedding_clusters: ndarray
        - word_clusters:
        - a: float -- alpha value
    Keyword Arguments:
        - filename: str -- default (default = None)

    Returns:
        None
    """
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters,
                                               word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=a, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right',
                         va='bottom', size=8)
    plt.legend(loc='best')
    plt.title(title)
    plt.grid(False)
    if filename:
        plt.savefig(filename, format='png', dpi=1500, quality=95)
    plt.show()


def fetch_clusters_3d(model):
    words_wp = []
    embeddings_wp = []
    for word in list(model.wv.vocab)[:5]:
        embeddings_wp.append(model.wv[word])
        words_wp.append(word)

    tsne_wp_3d = TSNE(perplexity=15, n_components=3, init='pca',
                      n_iter=3500, random_state=32, verbose=1)
    embeddings_wp_3d = tsne_wp_3d.fit_transform(embeddings_wp)
    return embeddings_wp_3d


def tsne_plot_3d(title, label, embeddings, filename, a=1):
    fig = plt.figure()
    ax = Axes3D(fig)
    colors = cm.rainbow(np.linspace(0, 1, 1))
    plt.scatter(embeddings[:, 0], embeddings[:, 1],
                embeddings[:, 2], c=colors, alpha=a, label=label)
    plt.legend(loc='best')
    plt.title(title)
    if filename:
        plt.savefig(filename, format='png', dpi=1500, quality=95)
    plt.show()


def plot_pca(model, top_n):
    vocab = model.wv.vocab
    # Get first n
    top = {k: vocab[k] for k in list(vocab)[:top_n]}

    # Plot PCA
    X = model[top]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    plt.scatter(result[:, 0], result[:, 1])

    # Add top annotations
    words = list(top)
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))

    # Plot
    plt.show()


def visualize_gridsearch(path, save_path):
    # load data
    with open(path, 'r') as f:
        data = json.load(f)

    score = pd.DataFrame(data).sort_values(by='correlation', ascending=False)
    size = score.shape[1]
    score = score.rename(
        columns={'min_count': 'min count', 'correlation': 'correlation'})
    q = np.concatenate((np.arange(size-2), np.arange(size-2)))
    d = score.drop(['correlation'], axis=1)

    f, axes = plt.subplots(
        size-2, size-1, figsize=(20, 10), squeeze=True, sharey=True)

    for c_ind, column_to_drop in tqdm(enumerate(list(d.columns))):
        data_ = d.drop(column_to_drop, axis=1)
        for index, val in enumerate(list(data_.columns)):
            a = sns.swarmplot(data=score,
                              x=column_to_drop,
                              y='correlation',
                              hue=val,
                              ax=axes[index, c_ind],
                              size=10)
            if c_ind != 0:
                a.get_yaxis().set_visible(False)
            a.tick_params(labelsize=20)
            a.set_ylabel('correlation', fontsize=20)
            a.set_xlabel(column_to_drop, fontsize=20)

    plt.subplots_adjust(hspace=0, wspace=0)
    plt.savefig(save_path, bbox_inches="tight")


if __name__ == "__main__":
    initialize_logger()

    (_, _, _, _, _, _, path, _) = directory_variables()

    model = KeyedVectors.load_word2vec_format(path, binary=False)

    # plot_pca(model, 15)

    # keys = ['time_bn:00077270n', 'united_states_bn:00003341n',
    #         'village_bn:00042729n', 'also_bn:00114246r',
    #         'say_bn:00093287v', 'film_bn:00034471n']
    # embeddings_en_2d, word_clusters = fetch_clusters_2d(
    #     model, keys, top_n=15)
    # tsne_plot_similar_words('Similar Sense Embeddings', keys,
    #                         embeddings_en_2d, word_clusters, 0.7,
    #                         'similar_words.png')

    json_path = r'C:/Users/Sheikh/Documents/GitLab/word_sense_embeddings/resources/grid_search.json'
    save_path = r'hs_sg_results_gridsearch.png'
    visualize_gridsearch(json_path, save_path)

