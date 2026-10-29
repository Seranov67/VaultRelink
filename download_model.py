"""Download the pinned official model files once; inference itself is local."""
from pathlib import Path
from huggingface_hub import hf_hub_download

target = Path(__file__).resolve().parent / 'models' / 'multilingual-minilm-avx2'
revision = 'e8f8c211226b894fcb81acc59f3b34ba3efd5f42'
for name in ('tokenizer.json', 'onnx/model_quint8_avx2.onnx'):
    print('Downloading', name, flush=True)
    hf_hub_download(repo_id='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                    revision=revision, filename=name, local_dir=target)
print('Ready:', target)
