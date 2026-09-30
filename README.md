# Bank Term Deposit Classification

Predicts whether a client will subscribe to a term deposit from a Portuguese bank's direct-marketing call campaign (41,188 records, 21 features, [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)).

**Approach:** feature encoding, RFECV feature selection (46 optimal features), logistic regression tuned via GridSearchCV, then rebalanced with `class_weight='balanced'` to fix the class-imbalance blind spot.

**Result:** 86% accuracy, and recall on the "will subscribe" class up from 43% to 91% after rebalancing, a trade that materially improves campaign targeting.

**Stack:** Python, pandas, scikit-learn (LogisticRegression, RFECV, GridSearchCV), matplotlib/seaborn.

Model artifacts (`pipeline_bank.pkl`, `finalized_model.sav`) are included; the raw dataset is not (download it from the UCI link above and place it as `bank.csv` in this folder to re-run the notebook).
