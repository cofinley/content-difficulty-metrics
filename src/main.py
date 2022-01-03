import os
import re
import csv
import argparse
from typing import List
from pathlib import Path

import pandas as pd
import numpy as np
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


def get_wpm(text: str, start: str, end: str) -> float:
    words = re.split(WORD_SPLIT_REGEX, text)
    num_words = len(words)
    time_diff = float(end) - float(start)
    time_diff_minutes = time_diff / 60
    wpm = num_words / time_diff_minutes
    return wpm


def calculate_median_wpm(cues: List[dict]) -> float:
    wpms = []
    for cue in cues:
        wpm = get_wpm(cue['text'], cue['start'], cue['end'])
        wpms.append(wpm)
    return np.median(wpms)


def process_installments(text_series: pd.Series) -> pd.Series:
    all_text = text_series.str.cat(sep=' ')
    sentences = sbd.get_sentences(all_text.strip())
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
        t = tqdm(total=9, position=1, leave=False)
        t.update(1)
        t.write('Parsing subtitle csv...')
        cues = parse_csv(content_title)
        t.update(2)
        t.write('Calculating median WPM...')
        median_wpm = calculate_median_wpm(cues)
        t.update(3)

        t.write('Getting sentences from cues...')
        sentences = get_sentences_from_cues(cues)
        t.update(4)
        t.write('Saving sentences...')
        save_sentences(sentences, content_title)
        t.update(5)

        t.write('Calculating sentence metrics...')
        sentence_metrics_df = sentence_metrics.calculate_sentence_metrics(sentences)
        t.update(6)
        t.write('Saving sentence metrics...')
        sentence_metrics.save_sentence_metrics(sentence_metrics_df, content_title)
        t.update(7)

        t.write('Calculating summary metrics...')
        summary_metrics_dict = summary_metrics.calculate_summary(sentence_metrics_df, content_title)
        t.update(8)
        summary_metrics_dict['median_wpm'] = median_wpm
        t.write('Saving summary metrics...')
        summary_metrics.save_summary_metrics(summary_metrics_dict, content_title)
        t.update(9)
        t.close()


if __name__ == '__main__':
    print('Starting')

    argparser = argparse.ArgumentParser(
        description='Calculate median frequency and WPM of directory of subtitles')
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
