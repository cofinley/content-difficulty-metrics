import youtube_dl


ydl_opts = {
    'ignoreerrors': True,
    'skip_download': True,
    'writesubtitles': True,
    'writeautomaticsub': True,
    'subtitlesformat': 'ttml',
    'max_downloads': 50,
    'nooverwrites': True,
    'playlistrandom': True,
    'outtmpl': '../data/subtitles/%(uploader)s/%(title)s -- %(id)s.%(ext)s',
    'encoding': 'utf-8'

}


def main(link: str, target_language_code: str):
    ydl_opts['subtitleslangs'] = [target_language_code]
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([link])


if __name__ == '__main__':
    import argparse

    argparser = argparse.ArgumentParser(
        description='Calculate median frequency and WPM of directory of subtitles')
    argparser.add_argument(
        'link',
        help='Youtube Link to download',
        action='store')
    argparser.add_argument(
        '--tl',
        help='Target language (2-char code)',
        action='store')
    args = argparser.parse_args()

    main(args.link, args.tl)
