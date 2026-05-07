# Phishing Email Detection with Flask

This project demonstrates a phishing email detector using machine learning and Flask.
It now includes both text and metadata features, plus an explanation view for predictions.

## Setup

1. Create a Python virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## Train the model

Run the training script to build the model artifact:

```powershell
python train.py
```

This creates a `model/` folder containing the trained pipeline.

## Run the Flask app

Start the web server:

```powershell
python app.py
```

Open `http://127.0.0.1:5000` in your browser to test email text input.

## Deploy on Vercel

1. Make sure your repository includes the trained model artifact at `model/phishing_pipeline.joblib`.
2. If you do not already have Vercel installed, run:
   ```powershell
   npm install -g vercel
   ```
3. Log in to Vercel:
   ```powershell
   vercel login
   ```
4. Deploy from the project root:
   ```powershell
   vercel --prod
   ```

This project uses a static frontend at `index.html` and a serverless prediction endpoint at `api/predict.py`.

## What changed

- `features.py` now extracts extra email metadata like URL count, exclamation use, digit frequency, uppercase ratio, and suspicious keywords.
- `train.py` builds a pipeline with both TF-IDF text features and metadata features.
- `app.py` loads the full pipeline and shows the strongest positive/negative feature signals.
- `data/phishing_emails.csv` includes more realistic phishing and legitimate examples.

## API

The app also exposes a JSON endpoint:

```powershell
curl -X POST http://127.0.0.1:5000/api/predict -H "Content-Type: application/json" -d '{"email_text": "..."}'
```

The API now returns an `explanation` object that lists the top contributing features.

## Notes

- Replace the sample dataset in `data/phishing_emails.csv` with a larger real phishing dataset for better performance.
- For production deployment, disable `debug=True` and use a production WSGI server such as Gunicorn or Waitress.
