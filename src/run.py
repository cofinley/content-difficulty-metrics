import os
import json
import re
import statistics
from typing import List, Tuple
from pathlib import Path

import srt
from lxml import etree
from nnsplit import NNSplit
from tqdm import tqdm

LINE_BREAK_REGEX = re.compile('\<br\s?\/?\>')
NEW_LINE_REGEX = re.compile('(\r?\n)+')
CUE_REGEX = re.compile('\[[^\]]+\]')
END_PUNCTUATION_REGEX = re.compile('(?<![\d\.])[\.\?\!](?![\d\.])')
WORD_SPLIT_REGEX = re.compile('[\s-]+')
TOKEN_SPLIT_REGEX = re.compile('\w+')
HTML_TAG_REGEX = re.compile('<\/?\w>')
MUSIC_REGEX = re.compile('.*♪.*')
LEADING_SPACES_DASHES_REGEX = re.compile('^[\s\-:]+|[\s\-]+$')

FREQ_LIST_DIR = os.path.join('data', 'frequency_lists')
SUBTITLES_DIR = Path('data', 'subtitles')
SUPPORTED_SUBTITLE_FILE_TYPES = ('.srt', '.ttml')
REALISTIC_MAX_SENTENCE_LENGTH = 100
HEADER = {
    'Title': 'title',
    'Subtitle Files': 'num_subtitle_files',
    '80th Percentile Word Frequency': 'freq_percentile_80',
    'Median WPM': 'wpm_median',
    'Median WPS': 'wps_median',
    'Sentences': 'num_sentences',
    'True Sentences': 'num_true_sentences'
}


class ContentDifficultyRating:
    def __init__(self, verbose, target_language, native_language, content_title=None):
        self.verbose = verbose
        self.native_language = native_language
        self.target_language = target_language
        self.content_title = content_title
        self.tl_frequency_index = self.get_tl_frequency_index()
        self.nl_frequency_set = self.get_nl_frequency_set()
        self.splitter = NNSplit.load(self.target_language)

    def get_tl_frequency_index(self):
        list_path = os.path.join(FREQ_LIST_DIR, self.target_language + '.json')
        print(os.getcwd())
        with open(list_path, encoding='utf-8') as f:
            l = json.load(f)
            return {word: i for i, word in enumerate(l)}

    def get_nl_frequency_set(self):
        list_path = os.path.join(FREQ_LIST_DIR, self.native_language + '.json')
        with open(list_path, encoding='utf-8') as f:
            return set(json.load(f))

    @staticmethod
    def parse_srt(text) -> List[srt.Subtitle]:
        subtitles = list(srt.parse(text))
        return subtitles

    @staticmethod
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

    def get_token_frequency(self, token, use_nl_borrow_words=False):
        freq = self.tl_frequency_index.get(token)
        if freq is not None:
            return freq
        if use_nl_borrow_words and token in self.nl_frequency_set:
            # Borrow word from native language, higher frequency used (lower number)
            return 0
        else:
            # Not found in frequency lists, lowest frequency used (higher number)
            return len(self.tl_frequency_index)

    @staticmethod
    def clean_subtitle_text(text: str) -> str:
        text = re.sub(CUE_REGEX, '', text)
        text = re.sub(NEW_LINE_REGEX, ' ', text)
        text = re.sub(LEADING_SPACES_DASHES_REGEX, '', text)
        text = re.sub(HTML_TAG_REGEX, '', text)
        text = re.sub(MUSIC_REGEX, '', text)
        text = text.strip()
        text = text.lower()
        return text

    def get_word_frequencies2(self, subtitles: List[srt.Subtitle], use_nl_borrow_words=False) -> int:
        frequency_values = []
        for subtitle in subtitles:
            if not subtitle.content:
                continue
            text = self.clean_subtitle_text(subtitle.content)
            tokens = [token.strip() for token in TOKEN_SPLIT_REGEX.findall(text)]
            frequency_values += [self.get_token_frequency(token, use_nl_borrow_words=use_nl_borrow_words) for token in tokens]
        return frequency_values

    def get_word_frequencies(self, text: str, use_nl_borrow_words=False) -> int:
        tokens = [token.strip() for token in TOKEN_SPLIT_REGEX.findall(text)]
        frequency_values = [self.get_token_frequency(token, use_nl_borrow_words=use_nl_borrow_words) for token in tokens]
        return frequency_values

    def get_wpms(self, subtitles: List[srt.Subtitle]) -> List[int]:
        num_words = 0
        wpms = []
        for subtitle in subtitles:
            if not subtitle.content:
                continue
            text = self.clean_subtitle_text(subtitle.content)
            words = re.split(WORD_SPLIT_REGEX, text)
            num_words = len(words)
            time_diff = subtitle.end - subtitle.start
            time_diff_minutes = time_diff.total_seconds() / 60
            wpm = num_words / time_diff_minutes
            wpms.append(wpm)
        return wpms

    def get_wpm(self, text, start, end) -> int:
        words = re.split(WORD_SPLIT_REGEX, text)
        num_words = len(words)
        time_diff = end - start
        time_diff_minutes = time_diff.total_seconds() / 60
        wpm = num_words / time_diff_minutes
        return wpm

    def segment_sentences(self, text: str) -> List[str]:
        splits = self.splitter.split([text])[0]
        return splits

    def get_words_per_sentence2(self, subtitles: List[srt.Subtitle]) -> Tuple[List[int], int, int]:
        words_per_sentence_values = []
        num_true_sentences_found = 0
        all_text = ''
        for subtitle in subtitles:
            if not subtitle.content:
                continue
            text = self.clean_subtitle_text(subtitle.content)
            all_text += ' ' + text

        all_text = all_text.strip()
        has_punctuation = re.search(END_PUNCTUATION_REGEX, all_text) is not None
        use_true_sentences = False
        if has_punctuation:
            sentences = re.split(END_PUNCTUATION_REGEX, all_text)
            if max(len(re.split(WORD_SPLIT_REGEX, sentence)) for sentence in sentences) < REALISTIC_MAX_SENTENCE_LENGTH:
                use_true_sentences = True
                num_true_sentences_found += len(sentences)
        if not use_true_sentences:
            sentences = self.segment_sentences(all_text)

        for sentence in sentences:
            sentence = str(sentence).strip()
            words = re.split(WORD_SPLIT_REGEX, sentence)
            words_per_sentence_values.append(len(words))
        return words_per_sentence_values, len(sentences), num_true_sentences_found

    def get_words_per_sentence(self, all_text: str) -> Tuple[List[int], int, int]:
        words_per_sentence_values = []
        num_true_sentences_found = 0

        all_text = all_text.strip()
        has_punctuation = re.search(END_PUNCTUATION_REGEX, all_text) is not None
        use_true_sentences = False
        if has_punctuation:
            sentences = re.split(END_PUNCTUATION_REGEX, all_text)
            if max(len(re.split(WORD_SPLIT_REGEX, sentence)) for sentence in sentences) < REALISTIC_MAX_SENTENCE_LENGTH:
                use_true_sentences = True
                num_true_sentences_found += len(sentences)
        if not use_true_sentences:
            sentences = self.segment_sentences(all_text)

        for sentence in sentences:
            sentence = str(sentence).strip()
            words = re.split(WORD_SPLIT_REGEX, sentence)
            words_per_sentence_values.append(len(words))
        return words_per_sentence_values, len(sentences), num_true_sentences_found


    def calculate2(self):
        stats = []
        include_nl_borrow_words = self.native_language is not None
        for content_title in tqdm(os.listdir(SUBTITLES_DIR)):
            if self.content_title is not None and content_title != self.content_title:
                continue
            print('Content:', content_title)
            word_frequency_values = []
            wpms = []
            words_per_sentence_values = []
            total_num_sentences = 0
            total_num_true_sentences = 0
            subtitle_filenames = [f for f in os.listdir(os.path.join(SUBTITLES_DIR, content_title)) if f.endswith(SUPPORTED_SUBTITLE_FILE_TYPES)]
            for i, subtitle_filename in enumerate(subtitle_filenames):
                subtitle_path = os.path.join(SUBTITLES_DIR, content_title, subtitle_filename)
                if self.verbose:
                    print(i+1, '-', subtitle_filename)
                if subtitle_filename.endswith('.ttml'):
                    subtitles = self.parse_ttml(subtitle_path)
                elif subtitle_filename.endswith('.srt'):
                    with open(subtitle_path, encoding='utf-8') as srt_file:
                        subtitles = self.parse_srt(srt_file.read())
                word_frequency_values.extend(self.get_word_frequencies(subtitles, include_nl_borrow_words))
                wpms.extend(self.get_wpms(subtitles))
                wps_values, num_sentences, num_true_sentences_found = self.get_words_per_sentence(subtitles)
                words_per_sentence_values.extend(wps_values)
                total_num_sentences += num_sentences
                total_num_true_sentences += num_true_sentences_found

            freq_percentile_80 = statistics.quantiles(word_frequency_values, n=10, method='inclusive')[-2]
            median_wpm = statistics.median(wpms)
            median_wps = statistics.median(words_per_sentence_values)
            stats.append({
                'title': content_title,
                'num_subtitle_files': len(subtitle_filenames),
                'freq_percentile_80': int(freq_percentile_80),
                'wpm_median': int(median_wpm),
                'wps_median': median_wps,
                'num_sentences': total_num_sentences,
                'num_true_sentences': total_num_true_sentences
            })
        return stats

    def extract_subtitles(self, subtitle_path) -> List[srt.Subtitle]:
        if subtitle_path.endswith('.ttml'):
            subtitles = self.parse_ttml(subtitle_path)
        elif subtitle_path.endswith('.srt'):
            with open(subtitle_path, encoding='utf-8') as srt_file:
                subtitles = self.parse_srt(srt_file.read())
        return subtitles

    def calculate(self):
        stats = []
        include_nl_borrow_words = self.native_language is not None
        for content_title in tqdm(os.listdir(SUBTITLES_DIR)):
            if self.content_title is not None and content_title != self.content_title:
                continue
            print('Content:', content_title)
            word_frequency_values = []
            wpms = []
            words_per_sentence_values = []
            total_num_sentences = 0
            total_num_true_sentences = 0
            subtitle_filenames = [f for f in os.listdir(os.path.join(SUBTITLES_DIR, content_title)) if f.endswith(SUPPORTED_SUBTITLE_FILE_TYPES)]
            for subtitle_filename in subtitle_filenames:
                subtitle_path = os.path.join(SUBTITLES_DIR, content_title, subtitle_filename)
                subtitles = self.extract_subtitles(subtitle_path)
                all_text = ''
                for subtitle in subtitles:
                    if not subtitle.content:
                        continue
                    text = self.clean_subtitle_text(subtitle.content)
                    if not text:
                        continue
                    all_text += text + ' '
                    word_frequency_values.extend(self.get_word_frequencies(text, include_nl_borrow_words))
                    wpms.append(self.get_wpm(subtitle.content, subtitle.start, subtitle.end))
                wps_values, num_sentences, num_true_sentences_found = self.get_words_per_sentence(all_text)
                words_per_sentence_values.extend(wps_values)
                total_num_sentences += num_sentences
                total_num_true_sentences += num_true_sentences_found

            freq_percentile_80 = statistics.quantiles(word_frequency_values, n=10, method='inclusive')[-2]
            median_wpm = statistics.median(wpms)
            median_wps = statistics.median(words_per_sentence_values)
            stats.append({
                'title': content_title,
                'num_subtitle_files': len(subtitle_filenames),
                'freq_percentile_80': int(freq_percentile_80),
                'wpm_median': int(median_wpm),
                'wps_median': median_wps,
                'num_sentences': total_num_sentences,
                'num_true_sentences': total_num_true_sentences
            })
        return stats


    @staticmethod
    def print_results(stats):
        print(','.join(HEADER.keys()))
        for stat in stats:
            print(','.join([str(stat.get(header_lookup)) for header_lookup in HEADER.values()]))


if __name__ == '__main__':
    import argparse

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

    c = ContentDifficultyRating(args.v, args.tl, args.nl, args.content_title)
    stats = c.calculate()
    c.print_results(stats)
