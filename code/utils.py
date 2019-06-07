import numpy as np
import logging
from tqdm import tqdm


def save_to_file(sentences_lists, output_file_path):
    """
    Takes a list of lists of the corpus sentences, and write
    them in a text output file.

    Arguments:
        sentences_lists {List of Lists}
        output_file_path {str}

    Returns:
        None
    """
    with open(output_file_path, encoding='utf-8', mode='w+') as output_file:
        for _list in tqdm(sentences_lists, desc="Saving parsed data"):
            output_file.write(str(_list) + '\n')


def append_to_file(sentences_lists, output_file_path):
    """
    Takes a list of lists of the corpus sentences, and append
    them in a text output file.

    Arguments:
        sentences_lists {List of Lists}
        output_file_path {str}

    Returns:
        None
    """
    with open(output_file_path, encoding='utf-8', mode='a') as output_file:
        for _list in tqdm(sentences_lists, desc="Appending parsed data"):
            output_file.write(str(_list) + '\n')


def initialize_logger():
    """
    Customize the logger, and fixes seed
    """
    np.random.seed(0)
    logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                        datefmt='%H:%M:%S', level=logging.INFO)


def process_huge_txt(path, NUM_OF_LINES=1_000_000):
    """Breaks huge file into smaller files of 5,000,000 lines
    each
    Arguments:
        path {[str]}
    Keyword Arguments:
        NUM_OF_LINES {[int]} (default: {1_000_000})
    """
    with open(path, encoding='utf-8', mode='r') as file:
        output = open("output1.txt", encoding='utf-8', mode="w")
        for i, line in tqdm(enumerate(file), desc='processing'):
            output.write(line)
            if i % NUM_OF_LINES == 0:
                output.close()
                output = open(f"output{int(i / NUM_OF_LINES + 1)}.txt",
                              encoding='utf-8', mode="w")
        output.close()
