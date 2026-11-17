# Fake News Detection System

[![CI](https://github.com/shxamill/fake_news/actions/workflows/ci.yml/badge.svg)](https://github.com/shxamill/fake_news/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Streamlit application that uses a trained text-classification model to estimate whether submitted news text resembles the examples labelled as fake or real during training. The app also attempts to retrieve related context from live news search and Wikipedia.

## Important limitation

This is an educational machine-learning project, not a reliable fact-checking authority. A model prediction or a search result does not prove that a claim is true or false. Verify important claims against independent, reputable primary sources before acting on them.

## Features

- Text input for a news article or social post.
- Text preprocessing before vectorization.
- A saved scikit-learn classifier and TF-IDF vectorizer.
- A supplementary attempt to find related news and reference material online.

External search depends on third-party services and network availability; it may return no results or irrelevant context.

## Tech stack

- Python
- Streamlit
- scikit-learn
- TF-IDF text features
- DuckDuckGo search and Wikipedia lookups

## Getting started

Use Python 3.11 or another version supported by the dependencies.

~~~bash
python -m venv .venv
~~~

Activate the environment:

~~~bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
~~~

Install dependencies:

~~~bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

Run the application from the repository root so the relative model paths resolve:

~~~bash
streamlit run app/app.py
~~~

Streamlit prints the local URL in the terminal.

## Repository structure

| Path | Purpose |
| --- | --- |
| app/app.py | Streamlit user interface and inference flow |
| model/fake_news_model.pkl | Trained classifier |
| model/tfidf.pkl | Fitted TF-IDF vectorizer |
| notebooks/ | Data preparation and model-training notebooks |
| requirements.txt | Python dependencies |

## Development checks

There is no dedicated automated test suite yet. A basic syntax check is:

~~~bash
python -m compileall -q app
~~~

When changing preprocessing or inference behavior, compare it with the preprocessing used to train the saved model and add tests before relying on the result.

## Model and data safety

Only load the committed pickle files when you trust their source. Python pickle deserialization can execute code; never load model files supplied by an untrusted person. Do not commit private datasets, credentials, or personal information.

## Contribution and support

See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), and [SUPPORT.md](SUPPORT.md).

## License

This project is licensed under the [MIT License](LICENSE).
