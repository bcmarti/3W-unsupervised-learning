#!/usr/bin/env bash
set -e

ENV_NAME="3W-unsupervised"

# 1. Criar environment conda
conda env create -f environment.yml

# 2. Ativar environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# 3. Upgrade básico do pip
pip install -U pip setuptools wheel

# 4. Instalar projetos SEM dependências
pip install -e ../3W --no-deps
pip install git+https://github.com/moment-timeseries-foundation-model/moment.git@main --no-deps
