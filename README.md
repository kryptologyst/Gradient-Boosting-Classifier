# Gradient Boosting Classifier

**Gradient Boosting** sequential ensemble with learning rate control and GridSearchCV tuning.

## Overview

- Three datasets: Wine, Iris, Breast Cancer
- Configurable boosting stages, learning rate, and tree depth
- Feature importance ranking
- GridSearchCV for hyperparameter optimization
- **Streamlit dashboard** with learning rate comparison

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
# CLI: python -m src.main train --dataset wine --lr 0.1 --tune
pytest tests/ -v
```

## Docker

```bash
docker compose up --build
```

## License

MIT
# Gradient-Boosting-Classifier
