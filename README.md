# IVFEngine
An interpretable deep learning-based system that uses clinical parameters from IVF treatment cycles to predict
live baby birth outcomes using FT-Transformer architecture
#Environment setup
# Install miniconda/anaconda first and then run the given command in linux terminal
conda create -n transformer python=3.8
conda activate transformer
pip install -r requirements.txt

# Data preprocessing
Prepare an XML-based latest spreadsheet (.xlsx) file to be used as an input. For all preprocessing steps, please take a look at the methodology section of our manuscript

# Quick start
cd translatomer3
conda activate transformer
streamlit run app.py
