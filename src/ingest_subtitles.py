import os
import re
import csv
import argparse
from typing import List
from pathlib import Path

import srt
from lxml import etree
from tqdm import tqdm

NEW_LINE_REGEX = re.compile(r'(\r?\n)+')
CUE_REGEX = re.compile(r'\[[^]]+]')
HTML_TAG_REGEX = re.compile(r'</?\w>')
MUSIC_REGEX = re.compile('.*♪.*')
LEADING_SPACES_DASHES_REGEX = re.compile(r'^[\s\-:]+|[\s\-]+$')

PROJECT_DIR = Path('..')
DATA_DIR = Path(PROJECT_DIR, 'data')
FREQ_LIST_DIR = Path(DATA_DIR, 'frequency_lists')
SUBTITLES_DIR = Path(DATA_DIR, 'subtitles')
CSV_DIR = Path(DATA_DIR, 'csv')
COMBINED_SUBTITLES_DIR = Path(CSV_DIR, 'combined_subtitles')

SUPPORTED_SUBTITLE_FILE_TYPES = ('.srt', '.ttml')
SUBTITLE_HEADER = ('main_title', 'content_title', 'text', 'start', 'end')


def create_dirs():
    COMBINED_SUBTITLES_DIR.mkdir(exist_ok=True, parents=True)


def parse_srt(text) -> List[srt.Subtitle]:
    subtitles = list(srt.parse(text))
    return subtitles


def parse_ttml(ttml_filepath):
    tree = etree.parse(ttml_filepath)
    root = tree.getroot()
    subtitles = []
    cues = root.find('body', root.nsmap).find('div', root.nsmap).findall('p', root.nsmap)
    for i, cue in enumerate(cues):
        content = ' '.join(cue.itertext())
        begin = cue.get('begin')
        start = srt.srt_timestamp_to_timedelta(begin)
        end = cue.get('end')
        end = srt.srt_timestamp_to_timedelta(end)
        subtitle = srt.Subtitle(index=i+1, start=start, end=end, content=content)
        subtitles.append(subtitle)
    return subtitles


def clean_subtitle_text(text: str) -> str:
    text = re.sub(CUE_REGEX, '', text)
    text = re.sub(NEW_LINE_REGEX, ' ', text)
    text = re.sub(LEADING_SPACES_DASHES_REGEX, '', text)
    text = re.sub(HTML_TAG_REGEX, '', text)
    text = re.sub(MUSIC_REGEX, '', text)
    text = text.strip()
    return text


def extract_subtitles(subtitle_path) -> List[srt.Subtitle]:
    if subtitle_path.endswith('.ttml'):
        subtitles = parse_ttml(subtitle_path)
    elif subtitle_path.endswith('.srt'):
        with open(subtitle_path, encoding='utf-8') as srt_file:
            subtitles = parse_srt(srt_file.read())
    return subtitles


def store_subtitles(rows: list, csv_path: str) -> None:
    with open(csv_path, 'w', newline='', encoding='utf-8') as o:
        writer = csv.writer(o)
        writer.writerow(SUBTITLE_HEADER)
        writer.writerows(rows)


def main(title: str = None):
    create_dirs()
    for main_title in tqdm(os.listdir(SUBTITLES_DIR)):
        if title is not None and main_title != title:
            continue
        print('Content:', main_title)
        csv_path = os.path.join(COMBINED_SUBTITLES_DIR, main_title + '.csv')
        subtitle_filenames = [f for f in os.listdir(os.path.join(SUBTITLES_DIR, main_title)) if f.endswith(SUPPORTED_SUBTITLE_FILE_TYPES)]
        rows = []
        for subtitle_filename in subtitle_filenames:
            content_title = re.split(rf'\.(?:{"|".join(SUPPORTED_SUBTITLE_FILE_TYPES).replace(".", "")})$', subtitle_filename)[0]
            subtitle_path = os.path.join(SUBTITLES_DIR, main_title, subtitle_filename)
            subtitles = extract_subtitles(subtitle_path)
            for subtitle in subtitles:
                if not subtitle.content:
                    continue
                text = clean_subtitle_text(subtitle.content)
                if not text:
                    continue
                subtitle.content = text
                rows.append((main_title, content_title, text, subtitle.start.total_seconds(), subtitle.end.total_seconds()))
        store_subtitles(rows, csv_path)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='Combine subtitles for content')
    argparser.add_argument(
        '--content-title',
        help='Content title matching subdirectory in data/subtitles'
    )
    args = argparser.parse_args()

    main(args.content_title)
