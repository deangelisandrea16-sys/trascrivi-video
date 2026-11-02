"""Legge richieste/*.txt (un link per file) e scrive trascrizioni/<nome>.md.
Nessun input passa dalla shell: yt-dlp e' usato come libreria."""
import os
import pathlib
import re
import tempfile

import yt_dlp
from faster_whisper import WhisperModel

REQ = pathlib.Path("richieste")
OUT = pathlib.Path("trascrizioni")
OUT.mkdir(exist_ok=True)
URL_RE = re.compile(r"^https://[A-Za-z0-9.-]+/\S*$")
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
MAX_SEC = 15 * 60


def ts(s: float) -> str:
    s = int(s or 0)
    return f"{s // 60:02d}:{s % 60:02d}"


def transcribe(url: str, cookie_file: str | None, model_box: list) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        opts = {
            "format": "bestaudio/best",
            "outtmpl": f"{tmp}/a.%(ext)s",
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "match_filter": yt_dlp.utils.match_filter_func(f"duration < {MAX_SEC}"),
        }
        if cookie_file:
            opts["cookiefile"] = cookie_file
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            path = ydl.prepare_filename(info)
        if not model_box:
            model_box.append(WhisperModel("small", device="cpu", compute_type="int8"))
        segs, meta = model_box[0].transcribe(path, vad_filter=True, beam_size=5)
        lines = [f"[{ts(s.start)}] {s.text.strip()}" for s in segs]
    author = info.get("uploader") or info.get("channel") or "Video"
    return "\n".join([
        f"# {author}",
        "",
        f"- Link: {url}",
        f"- Durata: {ts(info.get('duration'))}",
        f"- Lingua: {meta.language} ({meta.language_probability:.0%})",
        "",
        "## Didascalia",
        "",
        (info.get("description") or "(nessuna)").strip(),
        "",
        "## Trascrizione",
        "",
        *(lines or ["(nessun parlato rilevato)"]),
        "",
    ])


def main() -> None:
    cookie_file = None
    if os.environ.get("IG_COOKIES"):
        fd, cookie_file = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "w") as fh:
            fh.write(os.environ["IG_COOKIES"])
    model_box: list = []
    try:
        for req in sorted(REQ.glob("*.txt")):
            if not NAME_RE.match(req.stem):
                continue
            dest = OUT / f"{req.stem}.md"
            if dest.exists():
                continue
            words = req.read_text(encoding="utf-8").split()
            url = words[0] if words else ""
            if not URL_RE.match(url):
                dest.write_text("ERRORE: link non valido\n", encoding="utf-8")
                continue
            try:
                dest.write_text(transcribe(url, cookie_file, model_box), encoding="utf-8")
                print(f"OK {req.name}")
            except Exception as exc:  # noqa: BLE001
                msg = (str(exc).splitlines() or ["errore sconosciuto"])[0][:300]
                dest.write_text(f"ERRORE: {msg}\n", encoding="utf-8")
                print(f"ERRORE {req.name}: {msg}")
    finally:
        if cookie_file:
            os.remove(cookie_file)


if __name__ == "__main__":
    main()
