"""Streamlit test bench for the Lab 03 text generators."""
import csv
import io
import random
from pathlib import Path

import streamlit as st

from src.pipeline import HMM, NGramGenerator, coarse, generation_metrics, load_tagged, split_by_document

PLAIN, ROLE_SPLIT, TRIGRAM = "Plain HMM", "Role-split HMM", "Trigram"


@st.cache_resource
def get_models():
    corpus = Path(__file__).resolve().parent / "data" / "corpus.csv"
    train, _ = split_by_document(load_tagged(corpus))
    models = {
        PLAIN: HMM(train),
        ROLE_SPLIT: HMM(train, split_states=True),
        TRIGRAM: NGramGenerator([words for words, _ in train]),
    }
    return train, models


def generate_batch(train, models, model_name, temperature, seed, count):
    model = models[model_name]
    rng = random.Random(seed)
    patterns = {tuple(tags) for _, tags in train}
    samples, rows = [], []
    for _ in range(count):
        if model_name == TRIGRAM:
            words = model.sample_sentence(rng, temperature)
            states = models[ROLE_SPLIT].viterbi(words)
        else:
            words, states = model.sample_with_states(rng, temperature)
        tags = coarse(states)
        samples.append((words, tags))
        rows.append({
            "Sentence": " ".join(words),
            "Tag sequence": " ".join(states),
            "Valid pattern": tuple(tags) in patterns,
        })
    return rows, generation_metrics(samples, train, models[ROLE_SPLIT])


def to_csv(rows):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


st.set_page_config(page_title="Chaotic Text Generator", layout="wide")
st.title("Chaotic Text Generator")
st.caption("ENSET NLP · Lab 03")
st.info(
    "The corpus is synthetic and the grammar is tiny. Metrics such as valid tag pattern "
    "are defined relative to this corpus and do not measure the quality of real English."
)

st.session_state.setdefault("batch", None)
with st.sidebar:
    st.header("Generator settings")
    with st.form("generator"):
        model_name = st.selectbox("Model", (PLAIN, ROLE_SPLIT, TRIGRAM), index=1)
        temperature = st.slider("Temperature", 0.3, 3.0, 1.0, 0.1)
        seed = st.number_input("Random seed", min_value=0, value=42, step=1)
        count = st.number_input("Sentences", 10, 500, 50)
        submitted = st.form_submit_button("Generate", type="primary", width="stretch")

st.header("Generated sentences")
if submitted:
    train, models = get_models()
    rows, metrics = generate_batch(train, models, model_name, temperature, seed, count)
    st.session_state.batch = (rows, metrics, model_name, temperature, seed)

if st.session_state.batch is None:
    st.write("Choose a model and generate a batch to see its sentences and metrics.")
else:
    rows, metrics, batch_model, batch_temperature, batch_seed = st.session_state.batch
    st.caption(f"{batch_model} · {len(rows)} sentences · T = {batch_temperature:.1f} · Seed {batch_seed}")
    labels = {
        "valid_tag_pattern": "Valid tag pattern",
        "plausible_subject_verb": "Plausible subject–verb",
        "novel_sentences": "Novelty",
        "distinct_2": "Distinct-2",
    }
    for column, (key, label) in zip(st.columns(4), labels.items()):
        value = metrics[key]
        column.metric(label, f"{value:.3f}" if key == "distinct_2" else f"{value:.1%}", border=True)
    if batch_model == TRIGRAM:
        st.caption("Trigram tags are inferred by the role-split Viterbi tagger.")
    elif batch_model == ROLE_SPLIT:
        st.caption("Role states use S for subject, O for object, and P for prepositional phrase.")
    st.dataframe(rows, hide_index=True, height=290, alt="Generated sentences, tags and pattern validity")
    st.download_button("Download CSV", to_csv(rows), "generated_sentences.csv", "text/csv", on_click="ignore")
