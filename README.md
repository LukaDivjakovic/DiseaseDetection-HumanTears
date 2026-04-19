# DiseaseDetection-HumanTears
HackKosice project for UPJS

## How to run?

1. Create a virtual environment (python version 3.12)
2. Activate the virtual environment
3. Run `pip install -r requirements.txt`
4. Run `python solution.py`

## Evaluate saved EfficientNet models

Run evaluation on the same held-out test split used in `solution_efficientnet.py`.

```powershell
python evaluate_saved_model.py
python evaluate_saved_model.py --fold 1
python evaluate_saved_model.py --model-path models/final_model.keras
```
