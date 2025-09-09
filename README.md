## Hate Speech Detection with Fine-Tuned RoBERTa

This project fine-tunes `roberta-base` to detect hate speech using a labeled dataset (`HateSpeechDetection.csv`). It includes a Jupyter notebook for training, evaluation, and inference.

### Prerequisites
- Python 3.10–3.12 (Windows/macOS/Linux)
- Git (optional)

### Setup (recommended)
```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
# source .venv/bin/activate # macOS/Linux
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name hate_speech_llm --display-name "Python (hate_speech_llm)"
```

### Data
Place `HateSpeechDetection.csv` in the project root. The notebook expects columns:
- `Comment`: text field
- `Hateful`: binary label (0=Not Hateful, 1=Hateful)

### Run the notebook
1. Open `Hate_speech.ipynb` in Jupyter/VS Code.
2. Select kernel: `Python (hate_speech_llm)`.
3. Run all cells.

### Outputs
- Training/evaluation metrics printed during training
- Confusion matrix and ROC curve plots
- Saved model under `fine_tuned_roberta_hate_speech/<timestamp>/`

### Inference
The last notebook cell saves the model and provides a helper:
```python
predict_hate(["I love this community!", "You are disgusting and should be banned."])
```

### Common issues
- Windows safetensors file lock: saving uses `safe_serialization=False` to avoid locks.
- If downloads fail, ensure the internet connection or pre-cache the model:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
name = "roberta-base"
AutoTokenizer.from_pretrained(name)
AutoModelForSequenceClassification.from_pretrained(name)
```

### License
For academic/educational purposes. Review the dataset and model licenses before redistribution.


