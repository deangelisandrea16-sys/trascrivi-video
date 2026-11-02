# Trascrivi Video

Trascrive video da link (Instagram, TikTok, YouTube…) o da file.

- **Da link**: un file `richieste/<id>.txt` con il link fa partire GitHub Actions (yt-dlp + faster-whisper small). Il risultato va in `trascrizioni/<id>.md`.
- **Da file**: Whisper nel browser (Transformers.js), nessun upload.
- Pagina: GitHub Pages da `main` / root.
- Se Instagram blocca: Secret `IG_COOKIES` (cookies.txt formato Netscape di un account secondario).
- Nessun input passa dalla shell; Actions scrive solo in `trascrizioni/`.
