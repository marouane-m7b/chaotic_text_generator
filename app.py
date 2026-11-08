"""Streamlit test bench for the Lab 03 text generators."""
import csv
import io
import random
import re
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

generator_tab, tagger_tab = st.tabs(("Generate", "Viterbi tagger"))

with generator_tab:
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
        with st.container(horizontal=True):
            for key, label in labels.items():
                value = metrics[key]
                st.metric(label, f"{value:.3f}" if key == "distinct_2" else f"{value:.1%}", border=True, width=190)
        if batch_model == TRIGRAM:
            st.caption("Trigram tags are inferred by the role-split Viterbi tagger.")
        elif batch_model == ROLE_SPLIT:
            st.caption("Role states use S for subject, O for object, and P for prepositional phrase.")
        st.dataframe(
            rows,
            column_order=("Valid pattern", "Sentence", "Tag sequence"),
            column_config={
                "Valid pattern": st.column_config.CheckboxColumn(width="small", pinned=True),
                "Sentence": st.column_config.TextColumn(width="medium"),
            },
            hide_index=True,
            height=290,
            alt="Generated sentences, tags and pattern validity",
        )
        st.download_button("Download CSV", to_csv(rows), "generated_sentences.csv", "text/csv", on_click="ignore")

with tagger_tab:
    st.header("Viterbi tagger")
    st.caption("Decode the most probable part-of-speech tag for each word using the plain HMM.")
    sentence = st.text_input("Sentence to tag", placeholder="The cats watch the birds.", max_chars=1000)
    tokens = re.findall(r"\w+(?:['’-]\w+)*|[^\w\s]", sentence)
    if tokens:
        _, models = get_models()
        tagger = models[PLAIN]
        words = [token.lower() for token in tokens]
        unknown = list(dict.fromkeys(token for token, word in zip(tokens, words) if word not in tagger.vocab))
        if unknown:
            st.warning("Outside the training vocabulary: " + ", ".join(unknown))
        tags = tagger.viterbi(words)
        st.table([tokens, tags], hide_index=True, hide_header=True, alt="Words with their predicted part-of-speech tags underneath")
