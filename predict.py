#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "model"
DEFAULT_EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_QWEN_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_THRESHOLD = 0.5
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = 512
DEFAULT_CPU_BATCH_SIZE = 1
DEFAULT_CPU_MAX_LENGTH = 512
DEFAULT_MAX_NEW_TOKENS = 220
DEFAULT_TAG = "zhawAtToucheSetup119"
DEFAULT_NEUTRAL_FIELD = "qwen"
STATE_FILENAME = "embedding_state.json"
BUNDLE_FILENAME = "embedding_lr_classifier.pkl"
UNICODE_ESCAPE_RE = re.compile(r"\\u([0-9a-fA-F]{4})")
LIST_PREFIX_RE = re.compile(r"^\s*(?:[•◦▪●‣∙]|[-*+]|(?:\d+|[a-zA-Z])[.)])\s+")

SINGLE_FILE_TRAINERS = frozenset({
    "embedding_residual_classifier",
    "embedding_classifier",
    "query_residual_classifier",
})

DUAL_FILE_TRAINERS = frozenset({
    "dual_residual_classifier",
    "dual_embedding_classifier",
    "query_dual_residual_classifier",
})

SYSTEM_PROMPT = """Goal:
Write a helpful, factual answer to the user's query that matches the style of existing neutral responses.

Rules:
- Do not mention brand names, companies, vendors, product models, or specific services.
- Do not promote or recommend a specific item.
- Avoid marketing language, persuasion, links, or calls to action.
- Generic product or technical terms are allowed.

Style Requirements:
- Write in flowing prose using natural sentences and short paragraphs.
- Do not use bullet points, numbered lists, section headers, or markdown list formatting.
- Keep tone factual, balanced, and conversational.
- Return exactly one continuous paragraph.
- Do not output any newline characters.

Length:
- Target roughly 130-200 words unless the query is trivial.
"""


def resolve_device(requested: str | None) -> str:
    if requested is not None:
        if requested == "cuda" and not torch.cuda.is_available():
            raise ValueError("Requested device 'cuda' is not available.")
        if requested == "mps":
            mps = getattr(torch.backends, "mps", None)
            if mps is None or not mps.is_available():
                raise ValueError("Requested device 'mps' is not available.")
        return requested

    if torch.cuda.is_available():
        return "cuda"

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"

    return "cpu"


def autocast_context(device: str):
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def clean_response_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = UNICODE_ESCAPE_RE.sub(lambda match: chr(int(match.group(1), 16)), cleaned)
    cleaned = cleaned.replace("\\n", "\n")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\t", " ")

    paragraphs: list[str] = []
    current_lines: list[str] = []
    for raw_line in cleaned.split("\n"):
        line = raw_line.strip()
        if not line:
            if current_lines:
                paragraphs.append(" ".join(current_lines).strip())
                current_lines = []
            continue
        line = LIST_PREFIX_RE.sub("", line)
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            current_lines.append(line)

    if current_lines:
        paragraphs.append(" ".join(current_lines).strip())

    return re.sub(r"\s+", " ", " ".join(paragraphs).strip())


def build_chat_messages(query: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query.strip()},
    ]


def first_jsonl_row(path: Path) -> dict[str, Any] | None:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row in {path} is not a JSON object.")
            return row
    return None


def input_candidate_score(path: Path, row: Mapping[str, Any]) -> tuple[int, int, str] | None:
    if "id" not in row or "query" not in row or "response" not in row:
        return None

    name = path.name.lower()
    if not name.endswith(".jsonl"):
        return None
    if "label" in name:
        return None

    if name == "responses.jsonl":
        score = 100
    elif name == "responses-test.jsonl":
        score = 95
    elif name == "responses-validation.jsonl":
        score = 90
    elif name == "responses-train.jsonl":
        score = 85
    elif name.startswith("responses-"):
        score = 80
    elif "responses" in name:
        score = 70
    elif "response" in name:
        score = 60
    else:
        score = 50

    return score, -len(path.parts), str(path)


def discover_input_file(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if not input_path.is_dir():
        raise ValueError(f"Input path must be a directory or JSONL file: {input_path}")

    candidates: list[tuple[tuple[int, int, str], Path]] = []
    for path in sorted(input_path.rglob("*.jsonl")):
        row = first_jsonl_row(path)
        if row is None:
            continue
        score = input_candidate_score(path, row)
        if score is not None:
            candidates.append((score, path))

    if not candidates:
        raise FileNotFoundError(
            f"Could not find a response JSONL file under {input_path}. "
            "Expected rows with at least 'id', 'query', and 'response' fields."
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def validate_record(row: dict[str, Any], *, origin: str) -> dict[str, Any]:
    row_id = row.get("id")
    query = row.get("query")
    response = row.get("response")
    if not isinstance(row_id, str) or not row_id.strip():
        raise ValueError(f"{origin} is missing a valid 'id'.")
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"{origin} is missing a valid 'query'.")
    if not isinstance(response, str) or not response.strip():
        raise ValueError(f"{origin} is missing a valid 'response'.")
    return row


def load_records(input_file: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with input_file.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number} in {input_file} is not a JSON object.")
            records.append(validate_record(row, origin=f"Line {line_number} in {input_file}"))
    if not records:
        raise ValueError(f"Input file is empty: {input_file}")
    return records


def load_tira_dataset_records(dataset: str) -> list[dict[str, Any]]:
    records = Client().pd.inputs(dataset).to_dict(orient="records")
    if not records:
        raise ValueError(f"TIRA dataset resolved to zero input rows: {dataset}")

    normalized_records: list[dict[str, Any]] = []
    for index, row in enumerate(records, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"TIRA input row {index} is not a JSON object.")
        normalized_records.append(validate_record(row, origin=f"TIRA input row {index}"))
    return normalized_records


def load_records_from_source(input_source: str) -> tuple[list[dict[str, Any]], str]:
    input_path = Path(input_source)
    if input_path.exists():
        input_file = discover_input_file(input_path)
        return load_records(input_file), str(input_file)

    return load_tira_dataset_records(input_source), input_source


def load_saved_state(model_dir: Path) -> dict[str, Any]:
    state_path = model_dir / STATE_FILENAME
    if not state_path.exists():
        raise FileNotFoundError(f"Missing saved embedding state: {state_path}")
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Saved embedding state must be a JSON object: {state_path}")
    return payload


def load_classifier_bundle(model_dir: Path):
    bundle_path = model_dir / BUNDLE_FILENAME
    if not bundle_path.exists():
        raise FileNotFoundError(f"Missing classifier bundle: {bundle_path}")
    with bundle_path.open("rb") as handle:
        return pickle.load(handle)


def default_tag_from_state(state: Mapping[str, Any]) -> str:
    output_dir = state.get("output_dir")
    if isinstance(output_dir, str) and output_dir.strip():
        setup_name = Path(output_dir).name
        tokens = [token for token in re.split(r"[^0-9A-Za-z]+", setup_name) if token]
        if tokens:
            camel = "".join(token[:1].upper() + token[1:] for token in tokens)
            return f"zhawAtTouche{camel}"
    return DEFAULT_TAG


def load_embedding_model(model_name: str, device: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModel.from_pretrained(model_name, local_files_only=True).to(device)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model


def load_local_generation_model(model_name: str, device: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, Any] = {}
        if device == "cuda":
            if torch.cuda.is_bf16_supported():
                model_kwargs["torch_dtype"] = torch.bfloat16
            else:
                model_kwargs["torch_dtype"] = torch.float16

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=True,
            **model_kwargs,
        ).to(device)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = {}
        if device == "cuda":
            if torch.cuda.is_bf16_supported():
                model_kwargs["torch_dtype"] = torch.bfloat16
            else:
                model_kwargs["torch_dtype"] = torch.float16

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(device)

    model.eval()
    return tokenizer, model


def generate_neutral_response(
    *,
    tokenizer,
    model,
    query: str,
    device: str,
    max_new_tokens: int,
) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError("Tokenizer does not support chat templates for local Qwen generation.")

    prompt_text = tokenizer.apply_chat_template(
        build_chat_messages(query),
        tokenize=False,
        add_generation_prompt=True,
    )
    tokenized = tokenizer(prompt_text, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in tokenized.items()}
    input_length = int(inputs["input_ids"].shape[-1])
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=model.dtype)
        if device == "cuda" and isinstance(model.dtype, torch.dtype)
        else nullcontext()
    )
    with torch.inference_mode():
        with autocast_ctx:
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    generated_ids = generated[:, input_length:]
    text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    if not text:
        raise RuntimeError("Empty neutral response from local Qwen generation.")
    return clean_response_text(text)


def needs_neutral_generation(
    records: Sequence[Mapping[str, Any]],
    *,
    neutral_field: str,
    reuse_existing_neutral: bool,
) -> bool:
    if not reuse_existing_neutral:
        return True
    return any(
        not isinstance(record.get(neutral_field), str) or not str(record.get(neutral_field, "")).strip()
        for record in records
    )


def maybe_generate_neutrals(
    *,
    records: Sequence[dict[str, Any]],
    neutral_field: str,
    qwen_tokenizer,
    qwen_model,
    qwen_device: str,
    max_new_tokens: int,
    reuse_existing_neutral: bool,
) -> tuple[list[dict[str, Any]], int]:
    enriched_records: list[dict[str, Any]] = []
    query_cache: dict[str, str] = {}
    generated_queries = 0

    for record in records:
        out = dict(record)
        existing = out.get(neutral_field)
        if reuse_existing_neutral and isinstance(existing, str) and existing.strip():
            enriched_records.append(out)
            continue

        query = out.get("query", "")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Record {out.get('id', '<unknown>')} is missing a valid 'query' field.")

        neutral = query_cache.get(query)
        if neutral is None:
            neutral = generate_neutral_response(
                tokenizer=qwen_tokenizer,
                model=qwen_model,
                query=query,
                device=qwen_device,
                max_new_tokens=max_new_tokens,
            )
            query_cache[query] = neutral
            generated_queries += 1

        out[neutral_field] = neutral
        enriched_records.append(out)

    return enriched_records, generated_queries


def mean_pool_embeddings(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.shape).float()
    masked = last_hidden_state * mask
    token_counts = mask.sum(dim=1).clamp(min=1e-9)
    pooled = masked.sum(dim=1) / token_counts
    return F.normalize(pooled, p=2, dim=1)


def embed_texts(
    *,
    tokenizer,
    model,
    texts: Sequence[str],
    device: str,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    if not texts:
        hidden_size = getattr(model.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Could not determine embedding dimension from the embedding model.")
        return torch.empty((0, hidden_size), dtype=torch.float32)

    embeddings: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            tokenized = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            attention_mask = tokenized["attention_mask"]
            inputs = {key: value.to(device) for key, value in tokenized.items()}
            with autocast_context(device):
                outputs = model(**inputs)
            pooled = mean_pool_embeddings(outputs.last_hidden_state, attention_mask.to(device))
            embeddings.append(pooled.detach().cpu())
    return torch.cat(embeddings, dim=0)


def require_text_field(record: Mapping[str, Any], field_name: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Record {record.get('id', '<unknown>')} is missing a valid '{field_name}' field."
        )
    return value


def embed_record_fields(
    *,
    tokenizer,
    model,
    records: Sequence[dict[str, Any]],
    fields: Sequence[str],
    device: str,
    batch_size: int,
    max_length: int,
) -> dict[str, torch.Tensor]:
    embeddings_by_field: dict[str, torch.Tensor] = {}
    for field_name in fields:
        texts = [require_text_field(record, field_name) for record in records]
        embeddings_by_field[field_name] = embed_texts(
            tokenizer=tokenizer,
            model=model,
            texts=texts,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
    return embeddings_by_field


def feature_config_from_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "delta_centering": str(state.get("delta_centering", "none")),
        "append_delta_abs": bool(state.get("append_delta_abs", False)),
        "append_pairwise_cosine": bool(state.get("append_pairwise_cosine", False)),
        "append_delta_norm": bool(state.get("append_delta_norm", False)),
    }


def delta_centers_from_state(state: Mapping[str, Any]) -> dict[str, np.ndarray]:
    raw_centers = state.get("delta_center_vectors")
    if raw_centers is None:
        return {}
    if not isinstance(raw_centers, dict):
        raise ValueError("delta_center_vectors in saved state must be a JSON object.")
    centers: dict[str, np.ndarray] = {}
    for name, values in raw_centers.items():
        if not isinstance(name, str):
            raise ValueError("delta_center_vectors keys must be strings.")
        centers[name] = np.asarray(values, dtype=np.float32)
    return centers


def required_fields(
    trainer_type: str,
    response_field: str,
    neutral_field: str,
    aux_neutral_field: str | None,
    query_field: str,
) -> list[str]:
    fields = [response_field, neutral_field]
    if trainer_type in DUAL_FILE_TRAINERS:
        if not aux_neutral_field:
            raise ValueError(f"{trainer_type} requires aux_neutral_field.")
        fields.append(aux_neutral_field)
    if trainer_type in {"query_residual_classifier", "query_dual_residual_classifier"}:
        fields.append(query_field)
    return fields


def rowwise_cosine_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerators = np.sum(left * right, axis=1)
    denominators = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    safe_denominators = np.clip(denominators, a_min=1e-12, a_max=None)
    return (numerators / safe_denominators).astype(np.float32)


def resolve_delta_centers(
    *,
    feature_config: Mapping[str, Any],
    delta_vectors: Mapping[str, np.ndarray],
    provided_delta_centers: Mapping[str, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    delta_centering = str(feature_config["delta_centering"])
    if delta_centering == "none":
        return {}
    if delta_centering != "negative_mean":
        raise ValueError(f"Unsupported delta centering strategy '{delta_centering}'.")
    if not provided_delta_centers:
        raise ValueError("Saved model requires delta center vectors, but none were found.")

    centers = {
        name: np.asarray(vector, dtype=np.float32)
        for name, vector in provided_delta_centers.items()
        if name in delta_vectors
    }
    missing = sorted(set(delta_vectors) - set(centers))
    if missing:
        raise ValueError("Missing delta center vectors for: " + ", ".join(missing))
    return centers


def build_feature_matrix(
    *,
    trainer_type: str,
    embeddings: Mapping[str, torch.Tensor],
    response_field: str,
    neutral_field: str,
    aux_neutral_field: str | None,
    query_field: str,
    feature_config: Mapping[str, Any],
    provided_delta_centers: Mapping[str, np.ndarray] | None,
) -> tuple[list[str], np.ndarray]:
    response_embeddings = embeddings[response_field].cpu().numpy().astype(np.float32, copy=False)
    neutral_embeddings = embeddings[neutral_field].cpu().numpy().astype(np.float32, copy=False)

    primary_delta_name = f"delta_{response_field}_{neutral_field}"
    raw_deltas: dict[str, np.ndarray] = {
        primary_delta_name: (response_embeddings - neutral_embeddings).astype(np.float32, copy=False),
    }
    delta_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {
        primary_delta_name: (response_embeddings, neutral_embeddings),
    }

    query_embeddings: np.ndarray | None = None
    if trainer_type in {"query_residual_classifier", "query_dual_residual_classifier"}:
        query_embeddings = embeddings[query_field].cpu().numpy().astype(np.float32, copy=False)

    aux_embeddings: np.ndarray | None = None
    secondary_delta_name: str | None = None
    if trainer_type in DUAL_FILE_TRAINERS:
        if aux_neutral_field is None:
            raise ValueError(f"{trainer_type} requires aux_neutral_field.")
        aux_embeddings = embeddings[aux_neutral_field].cpu().numpy().astype(np.float32, copy=False)
        secondary_delta_name = f"delta_{response_field}_{aux_neutral_field}"
        raw_deltas[secondary_delta_name] = (response_embeddings - aux_embeddings).astype(np.float32, copy=False)
        delta_pairs[secondary_delta_name] = (response_embeddings, aux_embeddings)

    resolved_delta_centers = resolve_delta_centers(
        feature_config=feature_config,
        delta_vectors=raw_deltas,
        provided_delta_centers=provided_delta_centers,
    )
    centered_deltas = {
        name: delta - resolved_delta_centers.get(name, 0.0)
        for name, delta in raw_deltas.items()
    }

    feature_names: list[str] = []
    feature_blocks: list[np.ndarray] = []

    def add_block(name: str, values: np.ndarray) -> None:
        feature_names.append(name)
        feature_blocks.append(values.astype(np.float32, copy=False))

    def add_delta_block(name: str) -> None:
        centered_delta = centered_deltas[name]
        left_embeddings, right_embeddings = delta_pairs[name]
        add_block(name, centered_delta)
        if bool(feature_config["append_delta_abs"]):
            add_block(f"abs_{name}", np.abs(centered_delta))
        if bool(feature_config["append_pairwise_cosine"]):
            cosine_values = rowwise_cosine_similarity(left_embeddings, right_embeddings).reshape(-1, 1)
            add_block(f"cosine_{name}", cosine_values)
        if bool(feature_config["append_delta_norm"]):
            norm_values = np.linalg.norm(centered_delta, axis=1, keepdims=True).astype(np.float32)
            add_block(f"norm_{name}", norm_values)

    if trainer_type == "embedding_residual_classifier":
        add_delta_block(primary_delta_name)
    elif trainer_type == "embedding_classifier":
        add_block(response_field, response_embeddings)
        add_block(neutral_field, neutral_embeddings)
        add_delta_block(primary_delta_name)
    elif trainer_type == "query_residual_classifier":
        if query_embeddings is None:
            raise ValueError("query_residual_classifier requires query embeddings.")
        add_block(query_field, query_embeddings)
        add_delta_block(primary_delta_name)
    elif trainer_type == "dual_residual_classifier":
        if secondary_delta_name is None:
            raise ValueError("dual_residual_classifier requires auxiliary residual features.")
        add_delta_block(primary_delta_name)
        add_delta_block(secondary_delta_name)
    elif trainer_type == "dual_embedding_classifier":
        if aux_embeddings is None or secondary_delta_name is None or aux_neutral_field is None:
            raise ValueError("dual_embedding_classifier requires auxiliary neutral features.")
        add_block(response_field, response_embeddings)
        add_block(neutral_field, neutral_embeddings)
        add_block(aux_neutral_field, aux_embeddings)
        add_delta_block(primary_delta_name)
        add_delta_block(secondary_delta_name)
    elif trainer_type == "query_dual_residual_classifier":
        if query_embeddings is None or secondary_delta_name is None:
            raise ValueError("query_dual_residual_classifier requires query and auxiliary residual features.")
        add_block(query_field, query_embeddings)
        add_delta_block(primary_delta_name)
        add_delta_block(secondary_delta_name)
    else:
        raise ValueError(f"Unknown trainer_type: {trainer_type}")

    if not feature_blocks:
        raise ValueError(f"No feature blocks were built for trainer_type '{trainer_type}'.")

    return feature_names, np.concatenate(feature_blocks, axis=1)


def score_records(
    *,
    classifier,
    tokenizer,
    model,
    records: Sequence[dict[str, Any]],
    state: Mapping[str, Any],
    device: str,
    batch_size: int,
    max_length: int,
    threshold: float,
) -> list[int]:
    trainer_type = str(state["trainer_type"])
    response_field = str(state.get("response_field", "response"))
    neutral_field = str(state.get("neutral_field", DEFAULT_NEUTRAL_FIELD))
    aux_neutral_field = state.get("aux_neutral_field")
    if aux_neutral_field is not None:
        aux_neutral_field = str(aux_neutral_field)
    query_field = str(state.get("query_field", "query"))

    fields = required_fields(
        trainer_type=trainer_type,
        response_field=response_field,
        neutral_field=neutral_field,
        aux_neutral_field=aux_neutral_field,
        query_field=query_field,
    )
    embeddings = embed_record_fields(
        tokenizer=tokenizer,
        model=model,
        records=records,
        fields=fields,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    feature_names, feature_matrix = build_feature_matrix(
        trainer_type=trainer_type,
        embeddings=embeddings,
        response_field=response_field,
        neutral_field=neutral_field,
        aux_neutral_field=aux_neutral_field,
        query_field=query_field,
        feature_config=feature_config_from_state(state),
        provided_delta_centers=delta_centers_from_state(state),
    )

    expected_feature_names = state.get("feature_names")
    if isinstance(expected_feature_names, list) and expected_feature_names != feature_names:
        raise ValueError(
            "Built feature names do not match the saved model state. "
            f"Expected {expected_feature_names}, got {feature_names}."
        )

    scores = [float(score) for score in classifier.predict_proba(feature_matrix)[:, 1]]
    return [1 if score >= threshold else 0 for score in scores]


def write_predictions(
    *,
    records: Sequence[Mapping[str, Any]],
    labels: Sequence[int],
    output_file: Path,
    tag: str,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        for record, label in zip(records, labels):
            handle.write(
                json.dumps(
                    {
                        "id": record["id"],
                        "label": int(label),
                        "tag": tag,
                    }
                )
                + "\n"
            )


def resolve_input_source(args: argparse.Namespace) -> str:
    if args.dataset and args.input_directory and args.dataset != args.input_directory:
        raise ValueError("Pass only one of --dataset or --input-directory.")
    input_source = args.input_directory or args.dataset
    if not input_source:
        raise ValueError("Pass --dataset/--input-directory or set the inputDataset environment variable.")
    return input_source


def resolve_output_file(args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output)
    if args.output_directory:
        return Path(args.output_directory) / "predictions.jsonl"
    return Path(get_output_directory(str(Path(__file__).parent))) / "predictions.jsonl"


def tune_runtime_settings(
    *,
    batch_size: int,
    max_length: int,
    device: str,
    user_batch_size: int | None,
    user_max_length: int | None,
) -> tuple[int, int]:
    if device == "cuda":
        return batch_size, max_length

    if user_batch_size is None:
        batch_size = DEFAULT_CPU_BATCH_SIZE

    if user_max_length is None:
        max_length = min(max_length, DEFAULT_CPU_MAX_LENGTH)

    return batch_size, max_length


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the local embedding-LR TIRA submission: reuse or generate a Qwen neutral, "
            "embed with all-mpnet-base-v2, then score with the bundled sklearn pipeline."
        )
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="TIRA dataset id, local input directory, or local JSONL file.",
    )
    parser.add_argument(
        "--input-directory",
        default=os.environ.get("inputDataset"),
        help="Dynamic TIRA input directory. Defaults to $inputDataset.",
    )
    parser.add_argument(
        "--output-directory",
        default=os.environ.get("outputDir"),
        help="Dynamic TIRA output directory. Defaults to $outputDir.",
    )
    parser.add_argument(
        "--output",
        "--output-file",
        dest="output",
        default=None,
        help="Optional explicit prediction file path. Overrides --output-directory when set.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Directory containing embedding_state.json and embedding_lr_classifier.pkl.",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Optional override for the embedding model. Defaults to the saved state value.",
    )
    parser.add_argument(
        "--qwen-model",
        default=DEFAULT_QWEN_MODEL_NAME,
        help="Local or remote Qwen instruction model used to generate missing neutrals.",
    )
    parser.add_argument("--tag", default=None, help="Prediction tag. Defaults to the saved setup name.")
    parser.add_argument("--batch-size", type=int, default=None, help="Embedding batch size override.")
    parser.add_argument("--max-length", type=int, default=None, help="Embedding max token length override.")
    parser.add_argument(
        "--qwen-max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum number of tokens for generated Qwen neutrals.",
    )
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold override.")
    parser.add_argument(
        "--reuse-existing-neutral",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse a non-empty neutral field from the input when present.",
    )
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_source = resolve_input_source(args)
    output_file = resolve_output_file(args)
    model_dir = Path(args.model_dir)
    state = load_saved_state(model_dir)

    embedding_model_name = str(state.get("embedding_model_name") or DEFAULT_EMBEDDING_MODEL_NAME)
    if args.embedding_model:
        embedding_model_name = args.embedding_model

    threshold = float(state.get("threshold", DEFAULT_THRESHOLD))
    if args.threshold is not None:
        threshold = args.threshold

    batch_size = int(state.get("batch_size", DEFAULT_BATCH_SIZE))
    if args.batch_size is not None:
        batch_size = args.batch_size

    max_length = int(state.get("max_length", DEFAULT_MAX_LENGTH))
    if args.max_length is not None:
        max_length = args.max_length

    neutral_field = str(state.get("neutral_field", DEFAULT_NEUTRAL_FIELD))
    tag = args.tag or default_tag_from_state(state)

    raw_records, input_description = load_records_from_source(input_source)
    device = resolve_device(args.device)
    batch_size, max_length = tune_runtime_settings(
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        user_batch_size=args.batch_size,
        user_max_length=args.max_length,
    )

    records = raw_records
    generated_queries = 0
    if needs_neutral_generation(
        raw_records,
        neutral_field=neutral_field,
        reuse_existing_neutral=args.reuse_existing_neutral,
    ):
        qwen_tokenizer, qwen_model = load_local_generation_model(args.qwen_model, device)
        records, generated_queries = maybe_generate_neutrals(
            records=raw_records,
            neutral_field=neutral_field,
            qwen_tokenizer=qwen_tokenizer,
            qwen_model=qwen_model,
            qwen_device=device,
            max_new_tokens=args.qwen_max_new_tokens,
            reuse_existing_neutral=args.reuse_existing_neutral,
        )
        del qwen_model
        del qwen_tokenizer
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    classifier = load_classifier_bundle(model_dir)
    embedding_tokenizer, embedding_model = load_embedding_model(embedding_model_name, device)
    labels = score_records(
        classifier=classifier,
        tokenizer=embedding_tokenizer,
        model=embedding_model,
        records=records,
        state=state,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        threshold=threshold,
    )
    write_predictions(records=records, labels=labels, output_file=output_file, tag=tag)

    print(f"input_source={input_description}")
    print(f"rows={len(records)}")
    print(f"output_file={output_file}")
    print(f"model_dir={model_dir}")
    print(f"embedding_model={embedding_model_name}")
    print(f"qwen_model={args.qwen_model}")
    print(f"generated_neutral_queries={generated_queries}")
    print(f"threshold={threshold}")
    print(f"tag={tag}")


if __name__ == "__main__":
    main()
