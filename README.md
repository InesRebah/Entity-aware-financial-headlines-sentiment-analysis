# Financial Headlines Entity-Aware Sentiment Analysis

This repository contains the implementation and analysis for a project on **entity-aware sentiment analysis in financial news headlines**, based on the SEntFiN 1.0 dataset.

The goal is to predict the sentiment (positive, negative, neutral) expressed toward a **target entity** within a headline, especially in the presence of **multiple entities with potentially conflicting signals**.

- **notebook.ipynb**  
  Main experimentation notebook (EDA, feature engineering, model training, evaluation).

- **report_ines_rebah.pdf**  
  Final report detailing methodology, experiments, and conclusions.

- **sentfin_1.0.csv**  
  Dataset used in the project (entity-level annotated financial headlines).

- **src/**  
  Modular codebase:
  - `data/`: data loading and preprocessing
  - `features/`: feature engineering (TF-IDF, LSA, etc.)
  - `models/`: model implementations (Logistic Regression, XGBoost, BERT)
  - `error_analysis/`: evaluation tools and diagnostics

- **bert_results/**
  - `checkpoints/`: saved model checkpoints for BERT fine-tuning in the mixed standard framework

- **bert_results_single/**
  - `checkpoints/`: saved model checkpoints for BERT fine-tuning in the transfer framework
 
## Dataset License

This project uses the **SEntFiN 1.0 dataset**:

Sinha, A., Kedas, S., Kumar, R., Malo, P. (2021)

The dataset is distributed under the MIT License.  
See `LICENSE_DATASET.txt` for full details.
