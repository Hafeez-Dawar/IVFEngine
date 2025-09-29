# IVFEngine
An interpretable deep learning-based system that uses clinical parameters from IVF treatment cycles to predict
live baby birth outcomes using FT-Transformer architecture

## Environment Setup
This project was developed using a conda environment on a Linux system
Before runing the given command, please install miniconda/anaconda

Create conda environment
conda create -n transformer python=3.8

Activate environment
conda activate transformer

Install dependencies
pip install -r requirements.txt

## Data preprocessing
Prepare an XML-based latest spreadsheet (.xlsx) file to be used as an input. For all preprocessing steps, please take a look at the methodology section of our manuscript.

## Quick Start
cd translatomer3 
conda activate transformer

Launch the application
streamlit run app.py

# Citation
If you use IVFEngine in your research, please cite our work:
