import os


def directory_variables():
    """
    Sets all the environment variables for the whole program.

    Arguments:
        - None

    Returns:
        - file_path: str
        - output_file_path: str
        - mapping_file: str
        - filtered_corpus_output: str
        - model_path: str
        - tab_file_path: str
    """
    root_path = "../../../"
    root_dir = os.path.join(root_path, os.getcwd())

    file_path = os.path.join(root_dir, 'resources',
                             'EuroSense', 'eurosense.v1.0.high-precision.xml')

    train_o_matic_file_path = os.path.join(root_dir, 'resources',
                                           'train-o-matic-data',
                                           'EN', 'EN.500-2.0')

    sew_dir_path = r'D:/sew/sew_complete/*/*.xml'

    output_file_path = os.path.join(root_dir, 'resources',
                                    'EuroSense', 'corpus_output.txt')

    mapping_file = os.path.join(root_dir, 'resources', 'bn2wn_mapping.txt')

    filtered_corpus_output = os.path.join(
        root_dir, 'resources', 'EuroSense', 'filtered_corpus_output.txt')

    model_path = os.path.join(root_dir, 'resources', 'embeddings.vec')

    tab_file_path = os.path.join(
        root_dir, 'resources', 'wordsim353', 'combined.tab')

    return (file_path, train_o_matic_file_path, sew_dir_path,
            output_file_path, mapping_file,
            filtered_corpus_output, model_path, tab_file_path)


if __name__ == "__main__":
    (file_path, _, _, _, _, _, _, _) = directory_variables()
    print(file_path)
