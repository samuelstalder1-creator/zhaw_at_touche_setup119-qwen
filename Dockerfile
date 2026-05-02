FROM webis/touche25-ad-detection:0.0.1

ADD predict.py /predict.py
ADD requirements.txt /requirements.txt
ADD model /model

RUN pip3 install --no-cache-dir -r /requirements.txt

ARG EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
ARG QWEN_MODEL=Qwen/Qwen2.5-1.5B-Instruct

RUN python3 - <<PY
import json
import pickle
from pathlib import Path
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

embedding_model = "${EMBEDDING_MODEL}"
qwen_model = "${QWEN_MODEL}"

AutoTokenizer.from_pretrained(embedding_model)
AutoModel.from_pretrained(embedding_model)
AutoTokenizer.from_pretrained(qwen_model)
AutoModelForCausalLM.from_pretrained(qwen_model)

json.loads(Path("/model/embedding_state.json").read_text(encoding="utf-8"))
with Path("/model/embedding_lr_classifier.pkl").open("rb") as handle:
    pickle.load(handle)
PY

ENTRYPOINT ["python3", "/predict.py"]
