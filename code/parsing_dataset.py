import logging
import os
import re
import string
from glob import iglob

from lxml import etree
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from utils import initialize_logger, save_to_file, append_to_file

from directory_vars import directory_variables


def process_elements(elements, mapping_dict):
    """
    Takes in corpus elements and iterate over the annotation,
    and text tags. Skipping the empty tags. Then maps the BabelNet
    synsets to WordNet synsets.

    Tokenize the content of text tags, and creates annotation dictionary
    to map BabelNet synsets (fetched from annotation) to WordNet synsets.

    Arguments:
        - elements: lxml.etree._Element -- all sentence elements in corpus
        - mapping_dict: dict -- contain synsets only found in WordNet

    Returns:
        - sentences: list of list -- all sentences in corpus tokenized
        - annotation_dict: dict -- dictionary containing all the
            annotations in form of lemma_WordNetSynset
    """
    sentence, annotations_dict = [], dict()
    # Looping through the element
    for element in elements.iter():
        # Check for the element language (en), and not empty
        if element.get('lang') == 'en' and element.text is not None:
            if element.tag == 'text':
                # Tokenizing the sentences
                txt = process_text(element.text)
                sentence = word_tokenize(txt)
            elif element.tag == 'annotation':
                # Get the WordNet mapping for the element BabelNet Synset
                wn_mapping = mapping_dict.get(element.text)
                # If the BabelNet synset has a mapping, keep it
                # otherwise, just keep the word as it is.
                if wn_mapping is not None:
                    lemma = element.get('lemma').replace(' ', '_').lower()
                    val = f"{lemma}_{element.text}"
                    annotations_dict[element.get('anchor')] = val
    return sentence, annotations_dict


def get_mappings(mapping_file):
    """
    Opens mapping file passed to it, fetching all the BabelNet
    synsets and their correspondent WordNet synsets

    Arguments:
        - mapping_file: str -- points to the mapping file

    Returns:
        - mapping_dict: dict -- key:value >> bn_00005312a: 122453n
        - rev_mapping_dict: dict -- key:value >> 122453n: bn_00005312a 
    """
    logging.info(f"Reading the mappings from {mapping_file}")
    mapping_dict = dict()

    with open(mapping_file, mode='r') as file:
        # Read file contents
        lines = file.read().splitlines()
        # for every line split its content, to have key value pairs
        for line in tqdm(lines, desc="fetching mappings list"):
            line_arr = line.split('	')
            mapping_dict[line_arr[0]] = line_arr[1]

    rev_mapping_dict = {v: k for k, v in mapping_dict.items()}

    logging.info("Read the mappings.")
    return mapping_dict, rev_mapping_dict


def key_present(keys, key):
    """
    Checks if the key parameter passed is a substring of any
    of the keys list items.

    Arguments:
        - keys: list -- list of all the keys processed earlier
        - key: str

    Returns:
        - bool -- indicates whether key is present in keys or not
    """
    if len(key) == 0:
        return False
    for item in keys:
        if key in item:
            return True
    return False


def handle_sentence(sentence, annotations_dict):
    """
    Takes a tokenized sentence and annotations dict to map the words
    in the sentence to lemma_synset and make sure it has the words with
    longer length in term of keys, returns the input in lower case

    Example:
    Input: I believe in Gothic Cathedrals
    Output: I believe_bn:0054512v in Gothic_Cathedrals_bn:876452v

    ArgumentsL
        - sentence: list of str
        - annotations_dict: dict

    Returns:
        - sentence: list of str
    """
    # put sentence back in str format
    sentence_ = ' '.join(sentence)
    # fetch the keys from annotations dictionary
    keys = list(annotations_dict.keys())
    # Sort the list desc in order to have lengthier keys first
    keys.sort(key=len, reverse=True)
    # loop through keys to replace words with their lemma_synset
    for key in keys:
        # Check if key is in sentence and if the key does not stand
        # for unigram word while it can be replaced by its
        # longer lemma for better contextual meaning
        if key in sentence_ and not key_present(keys[:keys.index(key)], key):
            val = annotations_dict.get(key)
            sentence_ = sentence_.replace(key, val)
    # return tokenized sentence
    sentence_list = word_tokenize(sentence_)
    sentence_lower = [word.lower() for word in sentence_list]
    return sentence_lower


def read_dataset(file_name, output_file_path, mapping_dict):
    """
    Takes in file_name in xml to parse it, and iterate over
    sequence tags to get all the text & annotation elements
    of English language and saves the parsed data in a text
    output file.

    Arguments:
        file_name: {str} -- the path to xml file
        output_file_path: {str} -- the path for the parsed dataset

    Return:
        sentences_list: {List of lists} -- contains all the sentences in corpus
    """
    logging.info("Parsing dataset")
    # read file contents in terms of sentences
    context = etree.iterparse(file_name, tag='sentence')
    sentences_list = []
    # iterating over the sentences
    for _, elements in tqdm(context, desc="Parsing the corpus"):
        # Get from every element the sentence and annotations
        sentence, annotations_dict = process_elements(
            elements, mapping_dict)
        # if the sentence is not empty
        if len(sentence):
            # Map sentence words to their BabelNet synsets
            sentence_ = handle_sentence(sentence, annotations_dict)
            sentences_list.append(sentence_)
        elements.clear()
    logging.info("Parsed the dataset")

    return sentences_list


def process_trainomatic_sentence(elem, lemma, inv_dic):
    """Takes an elem and lemma, fetch from this elem
    a context and sense id in 'wn_xxxxxa' format, then
    replace sentence lemma with lemma_synset and returns
    this sentence

    Arguments:
        elem {[etree.Element]}
        lemma {[str]}

    Returns:
        [str] -- [sentence with lemma_synset pair]
    """
    ans, context = None, None
    for child in elem:
        if child.tag == 'answer':
            ans = child
        elif child.tag == 'context':
            context = child
    if (ans is not None) and (context is not None):
        wordnet_sense = ans.get('senseId').split(':')[1]
        synset = inv_dic.get(wordnet_sense)
        lemma_synset = f'{lemma}_{synset}'
        sentence = context[0]
        sentence = etree.tostring(
            sentence, method='text', encoding='UTF-8').decode().lower()
        sentence = sentence.replace(lemma, lemma_synset)
        return sentence


def process_trainomatic_elem(element, inv_dic):
    """For every element in train-o-matic file, if element is
    lexelt we fetch the lemma from it, else if it is an instance
    we get invoke process_trainomatic_sentence(elem, lemma) which
    is responsible to return a sentence with lemma_synset pairs.
    Tokenize sentence and return it

    Arguments:
        element {etree.Element}

    Returns:
        [list] -- [tokenized sentence]
    """
    lemma = None
    for elem in element.iter():
        if elem.tag == 'lexelt':
            lemma = elem.get('item').split('.')[0].lower()
        elif elem.tag == 'instance':
            sentence = process_trainomatic_sentence(elem, lemma, inv_dic)
            return word_tokenize(sentence)


def parse_train_o_matic(dir_path, inv_mapping_dict):
    """Parses train-o-matic dataset files

    Arguments:
        dir_path {str} -- [directory where all files exist]

    Returns:
        sentences_list [list of lists]
    """
    path = os.path.join(dir_path, '*.xml')
    sentences_list = []
    for xml_file in tqdm(iglob(path), desc="Parsing train-o-matic files"):
        context = etree.iterparse(xml_file, tag="corpus")
        for _, element in context:
            sentence = process_trainomatic_elem(element, inv_mapping_dict)
            if sentence is None:
                continue
            sentences_list.append(sentence)
            element.clear()
    return sentences_list


def remove_stopwords(txt):
    """Takes in txt tokenize it and remove all the stopwords using Regex

    Arguments:
        txt {str}

    Returns:
        [str] -- [same as the param passed but with no stopwords]
    """
    pattern = re.compile(
        r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    text = pattern.sub('', txt)
    return text


def process_text(txt):
    """Takes the text and remove punctuation, remove consecutive whitespaces,
    as well as, stop words, numbers, change text to lower case and returns it.

    Arguments:
        txt {str}
    Returns:
        [str]
    """
    txt = txt.translate(
        str.maketrans('', '', string.punctuation))
    remove_digits = str.maketrans('', '', string.digits)
    txt = s.translate(remove_digits)
    txt = re.sub(' +', ' ', txt)
    txt_ = txt.strip()
    txt_ = remove_stopwords(txt_)
    return txt_


def handle_sew_sentences(sentence):
    """
    Process the SEW sentences, splitting on new line, tokenizing,
    lowering the words (basic text preprocessing)

    Arguments:
        - sentence: str

    Returns:
        - list of lists
    """
    sentences_list = []
    if sentence is not None:
        sentences = sentence.split('\n')
    for sentence_ in sentences:
        splitted = word_tokenize(sentence_)
        sentences_lower = [word.lower() for word in splitted]
        if len(sentences_lower):
            sentences_list.append(sentences_lower)
    return sentences_list


def process_sew_elem(element, mapping_dict):
    """
    Process SEW XML elements to replace the text with proper
    annotations if they are in our mapping dictionary

    Arguments:
        - element: lxml.etree.Element
        - mapping_dict: dict

    Returns:
        - list of lists
    """
    sentence = None
    for elem in element.iter():
        if elem.tag == "text" and elem.text is not None:
            sentence = process_text(elem.text)
        elif elem.tag == "annotations":
            for child in elem:
                if child.text:
                    bn = child.xpath('babelNetID')[0].text
                    if mapping_dict.get(bn) is not None:
                        mention = process_text(child.xpath('mention')[0].text)
                        new_mention = '_'.join(mention.split())

                        old = r'\b{}\b'.format(mention)
                        new = f'{new_mention}_{bn}'
                        sentence = re.sub(old, new, sentence, count=1)

    sentences_list = handle_sew_sentences(sentence)
    return sentences_list


def parse_sew(dir_path, mapping_dict):
    """
    Parses SEW (Semantically Enriched Wikipedia) dataset files, which is extra
    huge, so every INTERVAL (number of files) parsed we save it to a file,
    for further processing

    Arguments:
        - dir_path: str -- points to path of the sew folders
        - mapping_dict: dict

    Returns:
        - list of lists
    """
    INTERVAL = 10000
    sentences_list = []
    for i, xml_file in tqdm(enumerate(iglob(dir_path)),
                            desc="Parsing SEW files"):
        if i % INTERVAL == 0:
            if i < INTERVAL:
                save_to_file(sentences_list, 'sew_parsed.txt')
            else:
                append_to_file(sentences_list, 'sew_parsed.txt')
            sentences_list.clear()
        context = etree.iterparse(
            xml_file, tag='wikiArticle', encoding='utf-8', recover=True)
        for _, element in context:
            sentences = process_sew_elem(element, mapping_dict)
            sentences_list.extend(sentences)
            element.clear()
    return sentences_list


def parse_datasets(file_name, train_o_matic_file_path, sew_dir_path,
                   output_file_path, mapping_file):
    """
    Opens the corpus to read its content and process it using mapping_dict
    and return (and save) list of tokenized sentences.

    Arguments:
        - file_name: str
        - output_file_path: str -- to save the corpus output
        - mapping_file: str -- get mapping from

    Returns:
        - sentences_list: list of lists
    """
    mapping_dict, inv_mapping_dict = get_mappings(mapping_file)
    sentences_list = read_dataset(file_name, output_file_path, mapping_dict)

    trainomatic_sentences = parse_train_o_matic(
        train_o_matic_file_path, inv_mapping_dict)
    sentences_list.extend(trainomatic_sentences)

    sew_sentences = parse_sew(sew_dir_path, mapping_dict)
    sentences_list.extend(sew_sentences)

    logging.info(f"Saving the dataset, length {len(sentences_list)}")
    # Save the list of sentences for later use, if needed
    save_to_file(sentences_list, output_file_path)
    logging.info(
        f"Dataset saved to {output_file_path}, in list of lists form.")

    return sentences_list


if __name__ == "__main__":
    initialize_logger()

    (file_path, train_o_matic_file_path, sew_dir_path,
     output_file_path, mapping_file, _, _, _) = directory_variables()

    sentences = parse_datasets(
        file_path, train_o_matic_file_path, sew_dir_path,
        output_file_path, mapping_file)
