# Content Difficulty Metrics

An older project from 2021 that I'm now archiving here for posterity/archiving.

This project attempted to derive the 'difficulty' of subtitled media using NLP. This was
used in hopes of automating the search for [comprehensible input](https://en.wikipedia.org/wiki/Input_hypothesis).
Using various metrics of the subtitles (vocab used and its frequency/commonality in the language, words per sentence,
words per minute, grammar structures per sentence, amount of proper noun usage, etc.), one could compare them between
media to get an estimate of relative difficulty.

This idea only went so far; not everything is subtitled and auto-generated subtitles don't have good sentence boundaries
on which the metrics rely. I transfer-learned [my own huggingface model](https://huggingface.co/cfinley/punct_restore_fr) (model construction repo [here](https://github.com/cofinley/punct_restore_fr)) to do sentence boundary detection (SBD) using
a corpus from opensubtitles. The model had good results, but even after collecting metrics, I didn't find that the
rankings between the media were accurate. You can see some cursory results that were brought into a Google Sheet [here](https://docs.google.com/spreadsheets/d/1F9YxGvBSHveR6llqxVc6Sor8PsFZ4iO_OUuNOvS12Og/edit?usp=sharing).

I learned that humans must be in the loop here. It would be better to derive a sequential ranking from some pairwise
comparisons of media (i.e. "Which is easier, Peppa Pig or Amélie?"). With enough crowdsourced data, I believe
subjectivity would diminish and great rankings would emerge. But that's for a different day.

**Note:** this was for French and has some hardcoded French things (like frequency list in sentence_metrics.py, SBD
model in sbd.py). I'm just committing this to put it out there for now.

## Process

Each step of the pipeline is self-contained in order to save intermediate states. Processing can take a lot of time and
this helps work more iteratively, so as to not lose all work at the end.

1. Get subtitles
    - Single file dropped into its own folder (e.g `data/subtitles/Amélie/Amélie.srt`)
    - Youtube channel
2. For each channel/TV series/movie:
    1. Combine subtitles for easier processing
    2. Extract sentences from subtitle cues
    3. Calculate difficulty metrics for each sentence
    4. Aggregate all sentence metrics into a single summary
3. Store/display summary metrics for further analysis

### Youtube Channel

`python src/youtube.py --tl TARGET_LANGUAGE_CODE LINK`

Link can be a channel, playlist, or video

### Combine Subtitles

`python src/ingest_subtitles.py [--content-title CONTENT_FOLDER_NAME]`

If content folder name specified with `--content-title` (relative to `data/subtitles`), only that folder's subtitles will be processed, otherwise process all folders in `data/subtitles`.
