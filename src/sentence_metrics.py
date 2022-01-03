"""
Transform/Metrics

Input: list of strings (sentences)
Output: dataframe (sentence metrics)
"""

import os
import json
import string
import argparse
from typing import List
from pathlib import Path

import pandas as pd
import numpy as np
import spacy
from spacy import symbols
from spacy.tokens import Token
from tqdm import tqdm

print('Loading spacy model for sentence metrics...')
nlp = spacy.load('fr_core_news_lg')
print('Finished loading spacy model.')

MAX_SENTENCES = 1000

PROJECT_DIR = Path('..')
DATA_DIR = Path(PROJECT_DIR, 'data')
FREQ_LIST_DIR = os.path.join(DATA_DIR, 'frequency_lists')
CSV_PATH = Path(DATA_DIR, 'csv')
SENTENCE_CSV_DIR = Path(CSV_PATH, 'sentences')
SENTENCE_METRICS_CSV_DIR = Path(CSV_PATH, 'sentence_metrics')


def create_dirs():
    CSV_PATH.mkdir(parents=True, exist_ok=True)
    SENTENCE_CSV_DIR.mkdir(exist_ok=True)
    SENTENCE_METRICS_CSV_DIR.mkdir(exist_ok=True)


TARGET_LANGUAGE = 'fr'
NATIVE_LANGUAGE = 'en'


def get_tl_frequency_index():
    list_path = os.path.join(FREQ_LIST_DIR, TARGET_LANGUAGE + '.json')
    print(os.getcwd())
    with open(list_path, encoding='utf-8') as f:
        l = json.load(f)
        return {word: i for i, word in enumerate(l)}


def get_nl_frequency_set():
    list_path = os.path.join(FREQ_LIST_DIR, NATIVE_LANGUAGE + '.json')
    with open(list_path, encoding='utf-8') as f:
        return set(json.load(f))


print('Loading frequency lists...')
tl_frequency_index = get_tl_frequency_index()
nl_frequency_set = get_nl_frequency_set()
print('Finished loading frequency lists.')


def get_token_frequency(token: str, use_borrow_words: bool = False):
    freq = tl_frequency_index.get(token)
    if freq is not None:
        return freq
    if use_borrow_words and token in nl_frequency_set:
        # Borrow word from native language, higher frequency used (lower number)
        return 0
    # Not found in frequency lists; not going to assume value; use out-of-vocab (OOV) sentinel value
    return -1


def get_word_frequencies(tokens: List[Token], use_borrow_words: bool = False) -> List[int]:
    lookups = []
    for token in tokens:
        if not token.text or token.text in string.punctuation:
            continue
        expanded_tokens = [t.replace("'", '').replace('-', '').lower() for t in token.text.split('-')]
        lookups.extend(t for t in expanded_tokens if t)

    frequency_values = []
    for token in lookups:
        freq = get_token_frequency(token, use_borrow_words)
        if freq != -1:
            frequency_values.append(freq)
    return frequency_values


def tree_height(root):
    """
    Find the maximum depth (height) of the dependency parse of a spacy sentence by starting with its root
    Code adapted from https://stackoverflow.com/questions/35920826/how-to-find-height-for-non-binary-tree
    :param root: spacy.tokens.token.Token
    :return: int, maximum height of sentence's dependency parse tree
    """
    if not list(root.children):
        return 1
    else:
        return 1 + max(tree_height(x) for x in root.children)


def calculate(df: pd.DataFrame):
    text = df['text']
    if not text:
        return None
    # Use nlp.pipe(texts, batch_size=N) for better performance, possibly with pd.concat
    doc = nlp(text)
    num_tokens = len(doc)
    mean_chars_per_token = np.mean([len(token) for token in doc])
    max_dep_tree_height = max([tree_height(token) for token in doc])
    num_propn = len([token for token in doc if token.pos == symbols.PROPN])
    num_entities = len(doc.ents)
    num_numbers = len([token for token in doc if token.pos == symbols.NUM])
    num_adj = len([token for token in doc if token.pos == symbols.ADJ])
    num_oov = len([token for token in doc if token.is_oov])
    num_noun_chunks = len(list(doc.noun_chunks))
    num_verbal_heads = len([
        possible_subject.head for possible_subject in doc
        if possible_subject.dep == symbols.nsubj and possible_subject.head.pos == symbols.VERB
    ])
    alpha_tokens = [token for token in doc if token.is_alpha]
    freq_values = get_word_frequencies(alpha_tokens, use_borrow_words=False)
    freq_values_with_borrow_words = get_word_frequencies(alpha_tokens, use_borrow_words=True)

    return (
        num_tokens,
        mean_chars_per_token,
        max_dep_tree_height,
        num_propn,
        num_entities,
        num_numbers,
        num_adj,
        num_oov,
        num_noun_chunks,
        num_verbal_heads,
        freq_values,
        freq_values_with_borrow_words
    )


def add_metrics_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        'num_tokens',
        'mean_chars_per_token',
        'max_dep_tree_height',
        'num_propn',
        'num_entities',
        'num_numbers',
        'num_adj',
        'num_oov',
        'num_noun_chunks',
        'num_verbal_heads',
        'freq_values',
        'freq_values_with_borrow_words'
    ]
    df['text'].replace('', np.nan, inplace=True)
    df.dropna(subset=['text'], inplace=True)
    sample_size = min(len(df), MAX_SENTENCES)
    df = df.sample(sample_size)
    df[columns] = df.progress_apply(calculate, axis='columns', result_type='expand')
    return df


def calculate_sentence_metrics(sentences: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame(sentences, columns=['text'], dtype=str)
    tqdm.pandas(unit='sentence', position=2)
    metrics_df = add_metrics_columns(df)
    return metrics_df


def save_sentence_metrics(sentence_metrics_df: pd.DataFrame, content_title: str) -> None:
    path = Path(SENTENCE_METRICS_CSV_DIR, content_title + '.csv')
    sentence_metrics_df.to_csv(path, index=False)


def main(content_title_override: str = None):
    create_dirs()
    for content_title in tqdm(os.listdir(SENTENCE_CSV_DIR)):
        content_title = content_title.split('.csv')[0]
        if content_title_override is not None and content_title != content_title_override:
            continue
        print('Content:', content_title)
        df = pd.read_csv(Path(SENTENCE_CSV_DIR, content_title + '.csv'))
        sentence_metrics_df = calculate_sentence_metrics(df['text'])
        save_sentence_metrics(sentence_metrics_df, content_title)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='Generate sentence metrics from content')
    argparser.add_argument(
        '--content-title',
        help='Content title matching subdirectory in data/csv/sentences'
    )

    args = argparser.parse_args()
    main(args.content_title)
