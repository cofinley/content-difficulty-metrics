"""
Medium metrics (summary of content metrics)

Input: dataframe (sentence metrics)
Output: dict (summary)
"""

import os
from pathlib import Path
from ast import literal_eval

import pandas as pd
from tqdm import tqdm

PERCENTILE = 0.8

PROJECT_DIR = Path('..')
DATA_DIR = Path(PROJECT_DIR, 'data')
CSV_DIR = Path(DATA_DIR, 'csv')
SENTENCE_METRICS_CSV_DIR = Path(CSV_DIR, 'sentence_metrics')
SUMMARY_METRICS_CSV_DIR = Path(CSV_DIR, 'summary_metrics')


def create_dirs():
    SENTENCE_METRICS_CSV_DIR.mkdir(exist_ok=True, parents=True)
    SUMMARY_METRICS_CSV_DIR.mkdir(exist_ok=True, parents=True)


def calculate_summary(sentence_metrics_df: pd.DataFrame, content_title: str) -> dict:
    """
    Calcuate means/medians of sentence metrics into single values
    """
    summary = {'content_title': content_title}

    series_columns = [
        'num_tokens',
        'max_dep_tree_height',
        'mean_chars_per_token',
        'num_propn',
        'num_entities',
        'num_numbers',
        'num_adj',
        'num_oov',
        'num_noun_chunks',
        'num_verbal_heads'
    ]
    reduction_columns = [
        'freq_values',
        'freq_values_with_borrow_words'
    ]

    p80_metrics = sentence_metrics_df[series_columns].quantile(PERCENTILE).to_dict()
    summary.update({'p80_' + k: v for k, v in p80_metrics.items()})

    for column in reduction_columns:
        p80 = pd.Series(sentence_metrics_df[column].sum()).quantile(PERCENTILE)
        summary['p80_'+column] = p80

    return summary


def save_summary_metrics(summary_metrics_dict: dict, content_title: str) -> None:
    path = Path(SUMMARY_METRICS_CSV_DIR, content_title + '.csv')
    df = pd.DataFrame([summary_metrics_dict])
    df.to_csv(path, index=False)


def main(content_title_override: str = None):
    create_dirs()
    total_df = pd.DataFrame()
    for content_title in tqdm(os.listdir(SENTENCE_METRICS_CSV_DIR)):
        content_title = content_title.split('.csv')[0]
        if content_title_override is not None and content_title != content_title_override:
            continue
        print('Content:', content_title)
        df = pd.read_csv(Path(SENTENCE_METRICS_CSV_DIR, content_title + '.csv'), converters={'freq_values': literal_eval, 'freq_values_with_borrow_words': literal_eval})
        summary_metrics_dict = calculate_summary(df, content_title)
        total_df = total_df.append(pd.DataFrame.from_dict([summary_metrics_dict]))
        save_summary_metrics(summary_metrics_dict, content_title)
    totals_path = Path(SUMMARY_METRICS_CSV_DIR, 'summary_totals.csv')
    total_df.to_csv(totals_path, index=False)


if __name__ == '__main__':
    import argparse

    argparser = argparse.ArgumentParser(
        description='Generate sentence metrics from content')
    argparser.add_argument(
        '--content-title',
        help='Content title matching subdirectory in data/csv/sentences',
        action='store')

    args = argparser.parse_args()
    main(args.content_title)
