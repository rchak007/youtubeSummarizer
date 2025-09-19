## info

LLsd laptop wsl  created project - 

/home/rchak007/youtubeV2

conda env - ytcap

Running it - 

streamlit run streamlit_app.py

Open the printed local URL (usually http://localhost:8501).
Paste your YouTube URL → click Show available subtitles to check language codes → set lang (often en) → Fetch → chunk → (optional) summarize.

sample - https://www.youtube.com/watch?v=bPI_YEt6RGQ


## info2 -
### Tip & Troubleshooting

 below-

- **Keep yt-dlp current**: `yt-dlp -U` if YouTube changes break things.
- **Captions missing?** Some videos have no subs → you’ll need the Whisper path (we can add as fallback later).
- **Token planning**: If you use summaries, keep chunk size reasonable (e.g., 2–3k words per chunk). You can tune the “Max words per chunk” control.
- **OpenAI key**: Put `OPENAI_API_KEY` in `.env`. Without it, the app runs but summary buttons won’t do anything.


URL:
youtubesummarizer-ai.streamlit.app