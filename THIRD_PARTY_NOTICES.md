# Preflight third-party resources

Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
Revision: e8f8c211226b894fcb81acc59f3b34ba3efd5f42
Upstream model card declares Apache-2.0.
https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

Uses upstream exported ONNX transformer, tokenizer, max length128 and attention-mask-aware mean pooling; cosine scores use normalized vectors.
Downloaded weights are excluded from source control. Runtime package versions are recorded in preflight/requirements-lock.txt; their upstream licences remain applicable.
