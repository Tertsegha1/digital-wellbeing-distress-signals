# Digital distress signals: language, sentiment and lifestyle markers under rigorous validation

Analysis code for the manuscript:

> Younas, S. & Anande, T. J. *Digital wellbeing in the social media era: dual pathway analysis of emotional and behavioural distress indicators and public health utility.* (Manuscript under submission, 2026.)

## Relationship to the main project

The main project repository is **[suzanayounas/Mental-Health-Vs-Social-Media](https://github.com/suzanayounas/Mental-Health-Vs-Social-Media)**. There, Suzana Younas conceived and first developed the dual-pathway analysis of emotional (Twitter) and behavioural distress indicators.

This repository contains the **methodological enhancements** by Tertsegha Joseph Anande that produce every result, table and figure reported in the manuscript:
- user-grouped (leakage-free) validation;
- VADER sentiment analysis with user-level and mixed-model tests;
- bootstrap confidence intervals;
- translation into population-level predictive values;
- the reproducible figure pipeline;
- re-analysis of the main project's *Social Media vs Productivity* dataset with the same rigorous pipeline.

The two repositories use the same Twitter corpus. The manuscript reports the behavioural pathway in two synthetic datasets: *Digital Habits vs. Mental Health*, and the main project's *Social Media vs Productivity* dataset, both analysed with the code in this repository.

## Overview

The study evaluates three widely reused open datasets within one reproducible framework:

- **Emotional pathway:** 20,000 English tweets from 72 Twitter users, labelled by whether the account belonged to a user identified as depressed. It uses TF-IDF with logistic regression and VADER sentiment, evaluated with a tweet-level split and with nested **user-grouped** cross-validation.
- **Behavioural pathway:** two **synthetic** datasets, with self-reported stress dichotomised at 6 or more and a random forest plus a logistic regression comparator:
  - *Digital Habits vs. Mental Health*: 100,000 records of screen time, sleep, TikTok use and number of platforms.
  - *Social Media vs Productivity* (the main project's dataset): 30,000 records of social media time, sleep, screen time before sleep, notifications and other digital habits, with demographics.

Main findings:
- **The language model's accuracy depends on how it is tested.** Its AUC is 0.835 with a tweet-level split but falls to 0.595 (95% CI 0.505–0.686) on users unseen in training, because the model largely recognises users and their topics.
- **Sentiment alone cannot identify unseen users** (AUC 0.539).
- **The two synthetic behavioural datasets give opposite answers.** In *Digital Habits* the models discriminate well (AUC 0.926 and 0.927), with effects far larger than those observed in real populations. In *Social Media vs Productivity* they perform at chance (AUC 0.506 and 0.498), even with every available variable, because stress is generated independently of behaviour. Synthetic results reflect how the data were generated.
- **At a population prevalence of 10%**, positive predictive values would be only 0.12 (language), 0.30 (*Digital Habits*) and 0.10 (*Social Media vs Productivity*).

## Repository structure

```
analysis/
  reanalysis.py          models, evaluation, bootstrap CIs, odds ratios, PPV tables
  emotional_visuals.py   word clouds, PCA landscape, LDA topics, 3D behaviour plot
  sentiment_analysis.py  VADER sentiment, user-level tests, mixed models, sentiment-only classifier
  behaviour_productivity.py  Social Media vs Productivity dataset: models, CIs, odds ratios, thresholds, subgroups
  calibration.py         calibration of every primary model (Brier, calibration-in-the-large, slope, plots)
  make_figures.py        manuscript figures from results.json
  scirep_figures.py      composite language figure (Scientific Reports layout)
  pipeline.dot           workflow diagram (Graphviz)
  output/                analysis outputs used in the manuscript
figures/                 all manuscript and supplementary figures
data/                    place the downloaded datasets here (not redistributed)
```

## Data

The datasets are not redistributed here. Download them from Kaggle and place them as shown in [`data/README.md`](data/README.md):

- *Depression: Twitter Dataset + Feature Extraction* (InFamousCoder, version 2): <https://www.kaggle.com/datasets/infamouscoder/mental-health-social-media>
- *Digital Habits vs. Mental Health* (Abhishek Dave, version 2; synthetic): <https://www.kaggle.com/datasets/abhishekdave9/digital-habits-vs-mental-health-dataset>
- *Social Media vs Productivity* (Mahdi Mashayekhi, version 1; synthetic): <https://www.kaggle.com/datasets/mahdimashayekhi/social-media-vs-productivity>

## Reproducing the results

Run all commands from the repository root.

```bash
pip install -r requirements.txt
```

1. **Models and statistics.** This step takes about 10–20 minutes and writes `analysis/output/`.
   ```bash
   python analysis/reanalysis.py --text data/twitter/Mental-Health-Twitter.csv --behaviour data/behaviour/digital_habits_vs_mental_health.csv --out analysis/output
   ```

2. **Descriptive language figures and the 3D behaviour plot.**
   ```bash
   python analysis/emotional_visuals.py
   ```

3. **Sentiment analysis.**
   ```bash
   python analysis/sentiment_analysis.py
   ```

4. **Social Media vs Productivity dataset.** This step takes about 10 minutes.
   ```bash
   python analysis/behaviour_productivity.py --data data/productivity/social_media_vs_productivity.csv
   ```

5. **Calibration.** This refits each model and checks it reproduces the published AUC before computing calibration.
   ```bash
   python analysis/calibration.py
   ```

6. **Remaining figures.**
   ```bash
   python analysis/make_figures.py --behaviour data/behaviour/digital_habits_vs_mental_health.csv
   python analysis/scirep_figures.py
   ```

7. **Workflow diagram.** This step needs Graphviz.
   ```bash
   dot -Tpng -Gdpi=300 analysis/pipeline.dot -o figures/fig1_pipeline.png
   ```

Tested with Python 3.11.9, scikit-learn 1.9.1, pandas 3.0.6 and statsmodels 0.15.0, with random seed 42. The full analysis was run twice and reproduced identically. Every number in the manuscript was checked automatically against these outputs.

## Outputs

| File | Contents |
|---|---|
| `results.json` | Model performance with 95% CIs, fold results, permutation importance, odds ratios, threshold sensitivity |
| `behaviour_descriptives.csv` | Behavioural descriptives by stress group with standardised mean differences |
| `behaviour_odds_ratios.csv` | Mutually adjusted odds ratios per unit of each behaviour |
| `ppv_population.csv` | Positive and negative predictive values and screening yield at 5–30% prevalence |
| `emotional_visuals.json` | Word-cloud vocabulary, PCA regions and LDA topic statistics |
| `sentiment.json` | VADER sentiment by group, user-level tests, mixed models, sentiment-only classifier |
| `productivity_results.json` | *Social Media vs Productivity*: model performance with 95% CIs, all-variables model, thresholds, permutation importance, odds ratios, subgroups |
| `productivity_descriptives.csv` | *Social Media vs Productivity* descriptives by stress group |
| `productivity_odds_ratios.csv` | *Social Media vs Productivity* mutually adjusted odds ratios |
| `ppv_population_productivity.csv` | *Social Media vs Productivity* predictive values at 5–30% prevalence |
| `calibration.json` | Calibration of every primary model: Brier and scaled Brier scores, calibration-in-the-large and slope, with 95% CIs |

`reanalysis.py` also writes `language_top_terms.csv`. That file is excluded from version control because it contains account-specific hashtags. Descriptive text analyses use only words written by at least five different users, so that individuals cannot be identified.

## Ethics

The study uses only publicly available, anonymised secondary data. No individual was contacted or re-identified, and no tweet text is reproduced.

## Acknowledgement of AI assistance

The analysis code was developed with the assistance of Claude (Anthropic). The authors reviewed all code and outputs and take full responsibility for them.

## Licence and citation

The code is released under the [MIT Licence](LICENSE). If you use it, please cite the article (see [`CITATION.cff`](CITATION.cff)).
