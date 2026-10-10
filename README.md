# AI-Based Plant Disease Detection 🌱

## Features
- Separate login and sign-up pages with farmer-themed background.
- AI image classification using a pretrained Hugging Face plant-disease model.
- English and Marathi language switch.
- Disease-specific basic care suggestions and agriculture support/shop categories.
- User-specific scan history and About page.
- Password hashing and SQLite database.

## Requirements
- Python 3.10 or newer recommended.
- Internet access on first model run (model weights download and may be large).
- A computer is recommended for running the AI model locally. Mobile-only hosting may not have enough memory.

## Run on Windows
1. Install Python 3.10+ from https://www.python.org/downloads/ and enable "Add Python to PATH".
2. Extract the ZIP.
3. Open Command Prompt in the extracted project folder.
4. Run:
   ```bash
   python -m venv .venv
   .venv\\Scripts\\activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   python app.py
   ```
5. Open http://127.0.0.1:5000 in your browser.
6. Create an account, sign in, then open Detect and upload a clear leaf photo.

## Run on macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```
Then visit http://127.0.0.1:5000.

## Important accuracy & safety notes
The configured model is a pretrained leaf image classifier (PlantVillage-style labels). It cannot guarantee correct results for every plant, every disease, blurry photos, or field conditions; its confidence score is not a guarantee. Treat results as preliminary screening only. Confirm with your local agriculture officer/KVK before using pesticides or removing valuable plants. Suggestions are general, not a substitute for crop-specific advice. Suggested shops are service categories, not live-verified store listings or endorsements.

## Before public deployment
Set a strong random `SECRET_KEY` environment variable, disable Flask debug mode, use HTTPS, configure secure cookies, add rate limiting and privacy/retention controls, and use a production WSGI server. This starter project is not yet hardened for public production.

## Updated report features
The report includes educational possible symptoms, possible causes, prevention/care, photo tips, and Google Maps search links. Enter your village/city before uploading to tailor nearby support searches. Verify distances, hours, contact details, and product availability on Maps/by phone. AI results are preliminary.
