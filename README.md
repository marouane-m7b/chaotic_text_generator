# Lab 03 · Chaotic Text Generator

An ENSET NLP test bench for comparing a plain POS-HMM, a role-split POS-HMM, and a trigram generator. Generate 10–500 sentences, inspect tag patterns and batch metrics, export CSV, or tag your own sentence with Viterbi.

## Run locally

From the repository root, using Python 3.12 or newer:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Evaluate

```bash
python -m src.evaluate
python -m src.evaluate_real
```

The first command writes the synthetic metrics and chart to `results/`. The second downloads Universal Dependencies English EWT on first run and writes the real-text metrics to the same directory. The EWT treebank is licensed under [CC BY-SA 4.0](https://github.com/UniversalDependencies/UD_English-EWT). The app uses the checked-in synthetic `data/corpus.csv`.

## Streamlit Community Cloud

Select repository `marouane-m7b/chaotic_text_generator`, branch `main`, and main file `app.py`. Use Python 3.12 in Advanced settings. No secrets are required. Open the resulting URL in a private window and test both tabs.
