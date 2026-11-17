# Contributing

This project combines a Streamlit interface, text preprocessing, serialized scikit-learn artifacts, and external search lookups.

## Local setup

~~~bash
python -m venv .venv
python -m pip install -r requirements.txt
streamlit run app/app.py
~~~

Activate the virtual environment before installing or running the application. Run Streamlit from the repository root because the app loads files from model/.

## Before opening a pull request

- Run python -m compileall -q app.
- Test empty input and a few ordinary text inputs.
- Check behavior when web search is unavailable.
- Keep inference preprocessing consistent with model training.
- Do not describe a prediction as proof that a claim is true or false.
- Do not commit credentials, private datasets, or user-submitted text.

There is currently no dedicated automated unit-test suite. New inference and preprocessing behavior should include tests.

## Model artifacts

Do not replace the pickle files unless the model and vectorizer are retrained and evaluated together. Record the training data source, preprocessing steps, dependency versions, evaluation methodology, and limitations when adding a new model.
