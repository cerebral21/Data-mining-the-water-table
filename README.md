# Data Mining the Water Table: Predicting Pump Status in Tanzania

Multiclass classification of water pump condition with CatBoost: **functional**, **functional needs repair** or **non functional**. This is the [DrivenData "Pump it Up: Data Mining the Water Table"](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/) competition data.

**Result:** about 77% accuracy (0.772 with the tuned model) on a 20% hold-out of the training data. See [Limitations](#limitations) for how much weight to put on that.

## Data

The competition provides three files: `training set values.csv`, `training set labels.csv` and `test set values.csv`. They are not included in this repo. Download them from the competition page and update the three paths in the second cell of the notebook. The notebook currently reads from `F:/water level/`.

<!--
TODO before publishing: add the majority-class baseline so the 77% has context.
Compute it with:  y.value_counts(normalize=True)
Then add a line to the Result section above, e.g.
"Always predicting the most common class gives XX% accuracy on the same split."
-->

## Approach

**Feature selection.** The raw data has about 40 columns. I kept a hand-picked subset of 14:

| Type | Columns |
|---|---|
| Numeric | `id`, `longitude`, `latitude` |
| Categorical | `basin`, `region_code`, `public_meeting`, `scheme_management`, `permit`, `extraction_type_class`, `management`, `payment_type`, `water_quality`, `quantity`, `source` |

**Missing values.** `permit`, `public_meeting` and `scheme_management` contain missing values; they are filled with an explicit `unknown` category rather than dropped or imputed.

**Encoding.** Categorical columns are converted to integer codes and declared to CatBoost as categorical features, so CatBoost applies its own ordered target encoding to them. The three status labels are mapped to 0 (functional), 1 (functional needs repair) and 2 (non functional).

**Model.** `CatBoostClassifier` with the `MultiClass` loss. An 80/20 random train/validation split (`random_state=0`). I compared learning rates (0.1, 0.2, 0.5, 0.7 and 1.0) and then tuned a final model: 1,000 iterations, learning rate 0.1, depth 6, `l2_leaf_reg=3`, `random_strength=2`.

| Model | Validation accuracy |
|---|---|
| CatBoost, 500 iterations, learning rate 0.1 | 0.766 |
| CatBoost, tuned (depth 6, `l2_leaf_reg=3`, `random_strength=2`) | 0.772 |

**Prediction.** The model predicts the competition test set and writes a submission CSV with the status names. I did not record a leaderboard score.

## Limitations

- **Validation set used for model selection.** CatBoost was given the same hold-out as its `eval_set` (which selects the best iteration by default), and I compared learning rates on it. The reported accuracy is therefore slightly optimistic.
- **Single split, no cross-validation.** One random split gives no estimate of variance.
- **Accuracy hides class balance.** "Functional needs repair" is a small class, and I did not report per-class precision and recall or a confusion matrix, so accuracy says little about how well that class is predicted.
- **`id` is used as a feature.** It is a row identifier and should be dropped.
- **Test set preprocessing differs from training.** The `unknown` fill is applied to the training data but not the test data, and the test categorical columns are encoded separately from the training set, so integer codes can differ between the two. This affects the submission, not the validation accuracy.
- **Hard-coded local paths** (`F:/...`) mean the notebook will not run as-is.

## Possible next steps

- Compute the majority-class baseline, per-class metrics and a confusion matrix.
- Replace the single split with stratified k-fold cross-validation, and keep a final test set untouched.
- Drop `id`; encode the training and test sets together (or pass raw strings to CatBoost) so the categories line up.
- Add engineered features from the columns I dropped (for example pump age from `construction_year`, and water-source and extraction details).

## Running it

```bash
pip install numpy pandas matplotlib seaborn scikit-learn catboost jupyter
jupyter notebook "data mining water table catboost.ipynb"
```

Set the three data paths in the notebook first.
