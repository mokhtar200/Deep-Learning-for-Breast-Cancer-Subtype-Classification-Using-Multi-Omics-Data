# DeepMO-BC: Deep Learning for Breast Cancer Subtype Classification Using Multi-Omics Data

## Overview

DeepMO-BC is a deep learning framework designed to classify breast cancer samples into PAM50 subtypes by integrating multiple omics data types from the TCGA-BRCA dataset.

## Features

- Integration of gene expression, DNA methylation, CNV, protein expression, and miRNA data.
- Modular deep learning architecture with late integration strategy.
- Comprehensive evaluation metrics and model interpretation using SHAP.

## Installation

```bash
git clone https://github.com/yourusername/deepmo-bc.git
cd deepmo-bc
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
