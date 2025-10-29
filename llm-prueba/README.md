# llm-prueba

Pequeño LLM autoregresivo en español (fine-tuning de `distilgpt2`) con recursos 100% gratuitos. Incluye scripts para preparar dataset, entrenar, generar ejemplos, subir a Hugging Face Hub y desplegar una demo en Spaces (Gradio).

## Requisitos
- Python 3.10+
- GPU opcional (Colab/Kaggle). CPU funciona pero más lento.

## Instalación rápida (local)
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 1) Preparar dataset
Por defecto usa un subconjunto pequeño de `opus_books` (lado español).
```bash
python src/data/prepare_dataset.py --source hf_opus_books --max_examples 50000 --output data/dataset.txt
```
Para usar archivos locales, colócalos en `data/raw/*.txt` y ejecuta:
```bash
python src/data/prepare_dataset.py --source local --output data/dataset.txt
```

## 2) Entrenamiento reproducible
```bash
python src/train.py \
  --model_name distilgpt2 \
  --dataset_path data/dataset.txt \
  --output_dir models/final_model \
  --epochs 1 \
  --block_size 128 \
  --batch_size 2 \
  --grad_accum_steps 8 \
  --save_steps 200 \
  --logging_steps 50 \
  --eval_ratio 0.05
```
Checkpoints y logs quedarán en `models/checkpoints` y `logs/`.

## 3) Generar ejemplos
```bash
python src/generate.py --model_dir models/final_model
```
Resultados en `outputs/examples.txt`.

## 4) Subir a Hugging Face Hub (modelo)
Necesitas un token con permiso `write`.
```powershell
$env:HF_TOKEN="<TU_TOKEN_HF>"
python src/upload_to_hub.py --model_dir models/final_model --repo_id <usuario>/llm-prueba-distilgpt2-es
```

## 5) Desplegar demo en Hugging Face Spaces (Gradio)
Crea un Space y sube `deploy/` automáticamente:
```powershell
$env:HF_TOKEN="<TU_TOKEN_HF>"
python deploy/deploy_space.py --space_repo <usuario>/llm-prueba-demo --model_repo <usuario>/llm-prueba-distilgpt2-es
```
Luego, en la configuración del Space, define la variable `MODEL_ID` con `<usuario>/llm-prueba-distilgpt2-es`.

## Colab
Abre `notebooks/colab_train_and_deploy.ipynb` en Google Colab. El notebook instala dependencias, monta Drive, prepara datos, entrena, genera ejemplos y opcionalmente sube a HF.

## Troubleshooting
- OOM: reduce `--block_size`, `--batch_size`, o aumenta `--grad_accum_steps`.
- Desconexión Colab: reanuda con checkpoints en `models/checkpoints`.
- Sin token HF: omite subida y sólo ejecuta hasta generación local.

## Acciones manuales necesarias
- Token de Hugging Face (write): https://huggingface.co/settings/tokens
- Usuario HF (para `repo_id`) y, opcional, crear un repo GitHub si deseas versionar el código.
- En Spaces, configurar `MODEL_ID` en Settings.
