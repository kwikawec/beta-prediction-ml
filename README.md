# Beta Prediction with Machine Learning

This repository contains the code accompanying the diploma thesis *Predikce ukazatele beta pomocí strojového učení* by **Ondřej Bouzek**, defended at the Faculty of Finance and Accounting, University of Economics and Business, Prague (VŠE Praha), 2026.

The code covers three parts of the empirical work:

- `beta_60m_prediction_pipeline.ipynb` – prediction pipeline for future 60-month realized beta,
- `beta_12m_prediction_pipeline.ipynb` – prediction pipeline for future 12-month realized beta,
- `shap_analysis.py` – SHAP analysis for the 60-month model specifications discussed in the thesis.

## Author

**Ondřej Bouzek**
Faculty of Finance and Accounting, University of Economics and Business, Prague,
Department of Corporate Finance

## Data

Input data are not included in this repository due to licensing restrictions (Bureau van Dijk Orbis). The scripts expect the input files to be placed in the `data/` directory. Empty CSV templates with the expected column names are provided in `data/templates/`.

Expected input files:

- `data/main_data_branch_60m.csv`
- `data/robustness_data_branch_60m.csv`
- `data/main_data_branch_12m.csv`
- `data/robustness_data_branch_12m.csv`
- `data/macro_country_fiscal_year_final.csv`

## Environment

The computations were prepared for Python 3.11. Install the package versions listed in `requirements.txt`:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

The neural-network results are sensitive to package versions, especially `scikit-learn`. The thesis results were prepared with `scikit-learn==1.5.0`, as specified in `requirements.txt`; other versions may produce slightly different predictions.

## Usage

Run the notebooks from the repository root after placing the required input files in `data/`.

\`\`\`bash
jupyter notebook beta_60m_prediction_pipeline.ipynb
jupyter notebook beta_12m_prediction_pipeline.ipynb
python shap_analysis.py
\`\`\`

Generated files are written to `outputs/`.

## Citation

If you use this code, please cite the underlying thesis:

> Bouzek, O. (2026). *Predikce ukazatele beta pomocí strojového učení.* Diploma thesis, University of Economics and Business, Prague.

## License

The code in this repository is released under the MIT License (see `LICENSE`). Input data are licensed separately by Bureau van Dijk and are not redistributable.

## Use

This repository is provided for academic review and non-commercial use in connection with the thesis. It is not a standalone public replication package because the underlying input data cannot be redistributed.
