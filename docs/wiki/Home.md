# Review Ratings Classifier — Project Wiki

NLP project with saved scikit-learn pipelines, a Flask prediction API and a static browser client. The ensemble pipeline uses review text; the SVM pipeline also uses a numeric rating.

## Quick start

```bash
python -m venv .venv
pip install -r requirements.txt
python app.py
```
Check `/health` before sending prediction requests. Use Python 3.10+ and compatible scikit-learn versions for the serialized pipelines.

## Key repository references

- [Source repository](https://github.com/aKmsdfhjb/REVIEW_RATINGS_CLASSIFIER)
- [README](../../README.md)
- [Architecture](Architecture.md)

## Scope and limitations

This wiki describes the repository as it exists today. A working static page does not imply that a Python API, model inference service, database, payment integration or external provider has also been deployed. Check the repository README and Actions history before treating a feature as production-ready.

## Documentation maintenance

Update this page when the implementation, dependencies, model artifacts or deployment architecture changes. Keep credentials, private user data and unlicensed datasets out of the repository.
