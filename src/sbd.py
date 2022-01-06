"""
Sentence Boundary Detection

Input: string (body)
Output: list of strings (sentences)
"""

import re
from typing import List

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

END_PUNCTUATION_REGEX = re.compile(r'(?<![\d.])[.?!](?![\w\d.])')
WORD_SPLIT_REGEX = re.compile(r'[\s-]+')
REALISTIC_MAX_SENTENCE_LENGTH = 100
BATCH_SIZE = 64

print('Loading custom huggingface model for sentence boundary detection (SBD)...')
tokenizer = AutoTokenizer.from_pretrained("cfinley/punct_restore_fr")

model = AutoModelForTokenClassification.from_pretrained("cfinley/punct_restore_fr")
nlp = pipeline('ner', model=model, tokenizer=tokenizer)
print('Finished loading SBD model.')


def get_sentences(body: str) -> List[str]:
    """
    Convert body of text into sentences
    Args:
        body: string body

    Returns: list of sentence strings
    """
    all_text = body.strip()
    has_punctuation = re.search(END_PUNCTUATION_REGEX, all_text) is not None
    use_true_sentences = False
    if has_punctuation:
        sentences = re.split(END_PUNCTUATION_REGEX, all_text)
        max_sent_length = max(len(re.split(WORD_SPLIT_REGEX, sentence)) for sentence in sentences)
        if max_sent_length < REALISTIC_MAX_SENTENCE_LENGTH:
            use_true_sentences = True
    if not use_true_sentences:
        sentences = segment_sentences(all_text)

    return [sentence.strip() for sentence in sentences]


def segment_sentences(text: str) -> List[str]:
    # Batches of 64 tokens through nlp()
    batch = []
    sents = []
    words = text.split()
    for word in words:
        if len(batch) == BATCH_SIZE:
            sents += process_batch(batch)
            batch = []
        batch.append(word)
    else:
        sents += process_batch(batch)
    return sents


def process_batch(word_batch: List[str]) -> List[str]:
    """
    Process batch of words into sentences
    Args:
        word_batch: batch of ``BATCH_SIZE`` words

    Returns: segmented sentences
    """
    sentences = []
    batch_text = ' '.join(word_batch)
    start_tags = nlp(batch_text)
    start_indexes = [start['start'] for start in start_tags]
    for i in range(len(start_indexes)):
        if i == 0:
            if start_indexes[i] == 0:
                continue
            elif len(sentences):
                # New batch starts off in the middle of a sentence, append to end of last sentence
                sent_tail = batch_text[:start_indexes[i]]
                sentences[-1] += ' ' + sent_tail
            continue
        sentence = batch_text[start_indexes[i - 1]:start_indexes[i]]
        sentences.append(sentence.strip())
        if i == len(start_indexes) - 1:
            sentence = batch_text[start_indexes[i]:]
            sentences.append(sentence.strip())
    return sentences
