# Review Ratings Classifier

An NLP project for classifying customer reviews with saved scikit-learn pipelines. The repository contains an ensemble pipeline and an SVM pipeline, plus a Flask API and browser interface for running predictions.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit--learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Flask](https://img.shields.io/badge/API-Flask-171715?style=flat-square&logo=flask&logoColor=white)

## Browser interface

The static interface is hosted separately from the Python inference API. After enabling **Settings → Pages → Build and deployment → GitHub Actions**, visit:

[Open Review Sentiment Lab](https://aKmsdfhjb.github.io/REVIEW_RATINGS_CLASSIFIER/)

Enter the URL of the deployed Flask API in the page before requesting a prediction.

## Models

- **Ensemble pipeline** — TF-IDF text features and a voting classifier, trained on review text.
- **SVM pipeline** — TF-IDF text features plus a scaled numeric rating, followed by a linear SVM.

The SVM endpoint therefore requires both a review and a rating. The ensemble endpoint uses review text only.

## Flowchart

~~~mermaid
flowchart TD
 A[Review text and optional rating] --> B{Selected model}
 B -->|Ensemble| C[TF-IDF text pipeline]
 B -->|SVM| D[TF-IDF plus rating preprocessor]
 C --> E[Voting classifier]
 D --> F[Linear SVM]
 E --> G[Predicted label]
 F --> G
 G --> H[Flask JSON response]
 H --> I[Browser result panel]
~~~

## Run locally

Use Python 3.10 or newer.

~~~bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
~~~

Open http://127.0.0.1:5000/health to check model availability. The static frontend can be opened locally and configured with http://127.0.0.1:5000.

## API

- GET /health — reports which model artifacts loaded.
- POST /predict — JSON body: {"review":"The product is excellent","model":"ensemble"}.
- For the SVM model, also provide "rating": 5.

## Project structure

- Model/ — serialized scikit-learn pipelines.
- Notebook/ — model development and evaluation notebooks.
- data.csv — source dataset.
- app.py — Flask inference API.
- index.html — static browser client.
- requirements.txt — runtime and training dependencies.
- .github/workflows/pages.yml — GitHub Pages deployment.

## Limitations

Model quality depends on the training data, label balance and the exact preprocessing used during training. Predictions are experimental and should not be treated as authoritative business decisions. Reinstall compatible scikit-learn versions if a serialized model cannot be loaded.

## License

MIT — see LICENSE.


## Project Wiki

- [Wiki home](docs/wiki/Home.md)
- [Architecture and deployment](docs/wiki/Architecture.md)

The repository keeps a version-controlled wiki source under `docs/wiki/`.
