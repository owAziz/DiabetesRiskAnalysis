# Implementation Plan: Diabetes risk model

## Overview

Build a reproducible classifier that predicts self-reported diabetes status (no diabetes, prediabetes, or diabetes) from the main inputs in `reports/contract.md`, using the 2024 CDC BRFSS public file. A second, separate summary answers what diabetes type people with diabetes report. Entry-level teammates each own one task at a time. Shared results live in Python modules and saved files. Notebooks are for exploration and error review only.

This is an analysis of survey self-report. It does not diagnose diabetes and it is not a clinical tool.

## How the team organizes the work

Use a small project, not one shared notebook.

| Place | What goes here | Who may edit it |
| --- | --- | --- |
| `src/diabetes_risk/` | Loading, cleaning, the split, training, and evaluation | The owner of the current task, reviewed by one teammate |
| `notebooks/` | Plots, questions, and error review | Only the author of that notebook |
| `data/raw/` | The CDC file, unchanged | Nobody edits it |
| `data/processed/` | The modeling table and the frozen split ids | Produced by a script |
| `reports/` | Tables, figures, and the short results write-up | The owner of the task that creates them |
| `models/` | The saved model and its metrics | Produced by the training command |

Rules that keep a junior team from blocking each other:

1. Agree the contract in Task 2 before anyone fits a model. The contract defines the target, the main columns, the sample, the metric, and the split. It does not fix the model type. After that, descriptive work and modeling can proceed on the same table.
2. One task has one owner. A second person reviews the change before the next task depends on it.
3. A notebook cell is a draft. If another person needs that result, move it into `src/diabetes_risk/` or save a file under `data/` or `reports/` the same day.
4. Everyone trains and scores on the split file from Task 6. Nobody draws a new random split inside a notebook.
5. Hand off work as files with column names, not as "run my notebook from the top."
6. Any number of people can work. Whoever is free takes the next task whose dependencies are done. One owner per task.

## Architecture Decisions

- **Target.** Three classes from `DIABETE4`: no diabetes, prediabetes, and diabetes. Interviews coded as diabetes only during pregnancy, don't know, refused, or blank stay out of the model. Pregnancy-only diabetes is a different condition from the three statuses in the project question.
- **Main model inputs.** `_AGE80` (age), `_BMI5` (BMI), `EXERANY2` (any exercise), `_SMOKER3` (cigarette smoking status), `USENOW3` (smokeless tobacco), `ECIGNOW3` (e-cigarettes), `_RFDRHV9` (heavy drinking), `SEXVAR` (sex), and `SSBSUGR2` (sugary soda). This is the main list, not a closed one. Another codebook column can be added on the contract page. Height and weight are descriptive columns. `_SMOKER3` is the four-level smoking status calculated from `SMOKE100` and `SMOKDAY2`.
- **Sample.** The modeling table is 40,000 interviews drawn from rows where sugary soda was asked and the diabetes target is one of the three classes, stratified on that target. The full public file stays in `data/raw/`.
- **Metric.** Choose the model with macro F1 on the validation split. Report per-class recall beside it. Record accuracy, and do not use it to pick the model: most respondents report no diabetes.
- **Split.** 60% train, 20% validation, 20% test, stratified on the target, one fixed seed. Selection happens on validation. The test set is scored once for the final write-up.
- **Models.** Model type is not fixed. A majority-class prediction is the reference. The chosen model is the one with the highest validation macro F1.
- **Survey weights.** Use the interview weight (expect `_LLCPWT`; confirm the name in the codebook) for the descriptive rates in Tasks 8 and 9. The classifier is scored as a predictor of interview records. The write-up says that plainly so nobody treats unweighted accuracy as a U.S. population rate.
- **Stack.** Python. Save the modeling table as Parquet. Read the SAS transport file once.

## Dependency Graph

```
Task 1 layout ─────────────┐
                           ├── Task 3 load thin table ── Task 4 dictionary
Task 2 contract ───────────┤                                    │
                           │                                    ├── Task 5 cleaning ── Task 6 split ── Task 10 models ── Task 11 comparison
                           │                                    │         │                                    │
                           │                                    │         ├── Task 7 quality                    ├── Task 12 error review
                           │                                    │         └── Task 8 relationships               └── Task 13 train command
                           └────────────────────────────────────┴── Task 9 diabetes type                              │
                                                                                                                       └── Task 14 results page
```

Tasks 8 and 9 can run at the same time as Task 10 once cleaning exists. Task 9 can start as soon as the thin table and dictionary exist.

## Task List

### Phase 1: Foundation

- [x] Task 1: Create the project layout
- [x] Task 2: Write the analysis contract

### Checkpoint: Foundation

- [x] The team has read the contract and agrees on the target, the exclusions, the main inputs, the 40,000-row sample, macro F1, and that model type is not fixed
- [x] Folders exist and raw data is listed in `.gitignore`

### Phase 2: Shared table

- [x] Task 3: Load BRFSS into a thin Parquet table
- [x] Task 4: Write the column dictionary
- [x] Task 5: Implement cleaning as importable functions
- [x] Task 6: Freeze the train, validation, and test ids

### Checkpoint: Shared table

- [ ] One command rebuilds the modeling table from `data/raw`
- [ ] The same split file is the only split the team uses
- [ ] Cleaning special codes are covered by tests

### Phase 3: Answer the question descriptively

- [x] Task 7: Data-quality notebook and saved summary
- [x] Task 8: Relationship charts for age, body size, and activity
- [x] Task 9: Diabetes-type table for respondents with diabetes

### Checkpoint: Descriptive answer

- [ ] Charts and the type table are files under `reports/`, not only notebook output
- [ ] Weighted rates say which weight column was used

### Phase 4: Model

- [x] Task 10: Fit models against a majority-class reference
- [x] Task 11: Compare models and name the chosen one
- [ ] Task 12: Error-analysis notebook on saved predictions
- [ ] Task 13: Single training command for the chosen model

### Checkpoint: Model

- [x] The chosen model is the highest validation macro F1, and the comparison says whether it beats the majority-class reference
- [ ] `python -m diabetes_risk.train` rewrites the model file and `reports/metrics.json`
- [ ] Test-set metrics are computed once and match the write-up

### Phase 5: Close

- [ ] Task 14: Results page with limitations

### Checkpoint: Complete

- [ ] A new teammate can run the training command from the README and get the same metrics
- [ ] The page states self-report, exclusions, the unused test set rule, and that this is not a diagnosis

## Risks and Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| The SAS file has 345 columns and 457,670 rows | High | Task 3 keeps the main columns and writes a 40,000-row Parquet sample |
| BRFSS uses 7, 9, 777, 999, and blank as non-answers | High | Dictionary plus tested cleaning functions; notebooks do not reimplement this |
| `HEIGHT3` and `WEIGHT2` mix feet/inches and meters in one field | High | `_BMI5` is the body-size input. Raw height and weight stay descriptive columns |
| `_BMI5` is often stored with implied decimals | Med | Dictionary confirms the scale; a test checks a known value |
| `DIABTYPE` is missing outside the diabetes module | Med | Task 9 restricts to respondents who were asked and reports how many that is |
| Class imbalance makes accuracy look strong | Med | Macro F1 and per-class recall are the decision metrics |
| Unweighted metrics get described as U.S. rates | Med | Weights are for descriptive tables only; the results page says so |
| Two people edit one notebook | Med | Notebooks stay personal; shared logic moves to `src/` |
| Gestational diabetes is folded into "yes" | Med | Contract excludes code for pregnancy-only diabetes |

## Open Questions

- Confirm in the 2024 codebook: the `DIABTYPE` code list, the `_BMI5` scale, and the weight variable name. `DIABETE4` codes are defined in `reports/contract.md`.

## Assumptions

1. The finished product is a classifier plus a short written answer, not a web app.
2. The label is self-reported `DIABETE4`, not a lab result.
3. Model type is not fixed. A majority-class prediction is the reference, and validation macro F1 names the chosen model.
4. There is no calendar. Order follows the dependencies above. Tasks 8 and 9 can proceed beside modeling after Task 5.
