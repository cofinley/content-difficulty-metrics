import os
import re
import csv
import argparse
from typing import List
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import sbd
import sentence_metrics
import summary_metrics
import ingest_subtitles

WORD_SPLIT_REGEX = re.compile(r'[\s-]+')

PROJECT_DIR = Path('..')
DATA_DIR = Path(PROJECT_DIR, 'data')
CSV_DIR = Path(DATA_DIR, 'csv')
COMBINED_SUBTITLES_CSV_DIR = Path(CSV_DIR, 'combined_subtitles')
SENTENCE_CSV_DIR = Path(CSV_DIR, 'sentences')
SENTENCE_METRICS_CSV_DIR = Path(CSV_DIR, 'sentence_metrics')
SUMMARY_METRICS_CSV_DIR = Path(CSV_DIR, 'summary_metrics')


def create_dirs():
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    COMBINED_SUBTITLES_CSV_DIR.mkdir(exist_ok=True)
    SENTENCE_CSV_DIR.mkdir(exist_ok=True)
    SENTENCE_METRICS_CSV_DIR.mkdir(exist_ok=True)
    SUMMARY_METRICS_CSV_DIR.mkdir(exist_ok=True)


def parse_csv(content_title) -> List[dict]:
    cues = []
    with open(Path(COMBINED_SUBTITLES_CSV_DIR, content_title + '.csv'), newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for line in reader:
            cues.append(line)
    return cues


def process_installments(text_series: pd.Series) -> pd.Series:
    all_text = text_series.str.strip().str.cat(sep=' ')
    sentences = sbd.get_sentences(all_text)
    return pd.Series(sentences)


def get_sentences_from_cues(cues: List[dict]) -> pd.Series:
    df = pd.DataFrame(cues)
    tqdm.pandas()
    sentences_by_installment = df.groupby(by='content_title')['text'].progress_apply(process_installments)
    # Reset index to remove hierarchical index
    sentences_by_installment = sentences_by_installment.reset_index(level=0)
    return sentences_by_installment


def save_sentences(sentences: pd.Series, content_title: str) -> None:
    path = Path(SENTENCE_CSV_DIR, content_title + '.csv')
    sentences.to_csv(path, index=False)


def main(target_language: str, native_language: str, is_verbose: bool, content_title_override: str = None):
    print('Creating dirs...')
    create_dirs()
    print('Ingesting subtitles...')
    ingest_subtitles.main(content_title_override)
    for content_title in tqdm(os.listdir(COMBINED_SUBTITLES_CSV_DIR), position=0, leave=False):
        if not content_title.endswith('.csv'):
            continue
        content_title = content_title.split('.csv')[0]
        if content_title_override is not None and content_title != content_title_override:
            continue
        print('Content:', content_title)
        cues = parse_csv(content_title)
        sentences = get_sentences_from_cues(cues)
        save_sentences(sentences, content_title)
        sentence_metrics_df = sentence_metrics.calculate_sentence_metrics(sentences)
        sentence_metrics.save_sentence_metrics(sentence_metrics_df, content_title)
        summary_metrics_dict = summary_metrics.calculate_summary(sentence_metrics_df, content_title)
        summary_metrics.save_summary_metrics(summary_metrics_dict, content_title)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='Calculate difficulty metrics of a directory of subtitles')
    argparser.add_argument(
        '--content-title',
        help='Content title matching subdirectory in data/subtitles',
        action='store')
    argparser.add_argument(
        '--tl',
        help='Target language (2-char code)',
        action='store')
    argparser.add_argument(
        '--nl',
        help='Native language (2-char code)',
        action='store')
    argparser.add_argument(
        '-v',
        help='Verbose mode',
        action='store_true')
    args = argparser.parse_args()

    main(args.tl, args.nl, args.v, args.content_title)
