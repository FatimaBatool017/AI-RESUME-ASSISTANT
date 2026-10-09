# AI-RESUME-ASSISTANT
# 📄 AI Resume ATS Checker

A Streamlit app that scores a resume for Applicant Tracking Systems (ATS) and suggests concrete improvements, powered by Google's Gemini Flash model.

## Features
- Upload a resume as **PDF, DOCX or TXT**
- Overall **ATS score (0–100)** with a breakdown by category: formatting, keywords, experience, structure, language
- Optional **job description** input for a job-match score and tailored missing keywords
- Strengths, missing keywords and prioritised (High / Medium / Low) improvements with example rewrites
- Download the full report as JSON

## Project structure
```
.
├── app.py             # Streamlit app
├── requirements.txt   # Python dependencies
└── README.md
```

## Run locally
1. Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey).
2. Install and run:
   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Provide the key (pick one):
   - Create `.streamlit/secrets.toml`:
     ```toml
     GEMINI_API_KEY = "your-key-here"
     ```
   - or set an environment variable (`export GEMINI_API_KEY=your-key-here`)
   - or paste it into the app's sidebar
4. Start the app:
   ```bash
   streamlit run app.py
   ```

## Configuration
| Name | Required | Description |
|------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Your Google Gemini API key |
| `GEMINI_MODEL` | No | Model name (default: `gemini-3.8-flash`) |

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (**never commit your API key**).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app**, choose the repo, branch `main` and main file `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
5. Click **Deploy**.

## Notes & limitations
- The ATS score is an AI-based estimate, not the result of any specific ATS product.
- Scanned/image-only PDFs can't be read; use a text-based PDF or DOCX.
- Resume text is sent to the Gemini API for analysis. Don't upload sensitive documents you can't share.
