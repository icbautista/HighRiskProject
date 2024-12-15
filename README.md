# Detecting Suicidal Ideation and Attempts in Clinical Notes

This repository is part of a project focused on analyzing clinical notes to identify instances of **Suicidal Ideation (SI**) and **Suicide Attempts (SA)**. It uses datasets from **MIMIC-III** and leverages Python scripts to preprocess data, extract relevant information, and organize it into a structured format suitable for further analysis.


---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Pipeline Workflow](#pipeline-workflow)
5. [How to Run the Scripts](#how-to-run-the-scripts)
6. [Installation Guide](#installation-guide)
---

## Project Overview

This project seeks to detect and classify SI and SA events by processing electronic health records (EHRs). Using text-sectionization methods and a pre-trained **RoBERTa model**, we aim to enhance the understanding of the textual data contained in clinical notes and support healthcare professionals in decision-making.

---

## Dataset

The project uses:
- **MIMIC-III Dataset**: A publicly available critical care dataset with de-identified health records of over 40,000 patients. (Access requires completing [PhysioNet’s Data Use Agreement](https://physionet.org/policy/PhysioNetDataUseAgreement.html)).
- **ScAN Dataset**: Used for expert-provided annotations to train the ScANER pipeline.
- Clinical notes undergo a thorough preprocessing step to extract the sections relevant to SI and SA.

---

## Project Structure

- **`data/`**: This folder contains limited input and output datasets, annotation files, and processed notes.
- **`scripts/`**: Holds the Python scripts designed for data processing and analysis.
  - `get_HADM_files.py`: Parses and arranges EHR notes into a structured format.
  - `Preprocess-Annotations-for-ScAN-predictor.py`: Prepares annotation data following the ScAN guidelines.
  - `Prediction_module_model.py`: Implements prediction models to classify SI/SA instances from clinical text.

---

## Pipeline Workflow

### Step 1: Data Preparation and Preprocessed Annotations
- Sectionization: Execute get_HADM_files.py to process raw MIMIC-III clinical notes, sectionizing them according to predefined guidelines that are essential for in-depth analysis.
- Annotation Alignment: Run Preprocess-Annotations-for-ScAN-predictor.py to match and align the sectionized   notes with the corresponding expert annotations. This prepares the dataset for training by linking text spans to their respective SI/SA classifications.

### Step 2: Classification Model Training and Evaluation
- Model Training: Use Prediction_module_model.py to train a RoBERTa-based neural network model. The script will transmit textual features through the model, adjusting its parameters to best fit the SA/SI detection task.
- Evaluation: After training, the same script will assess model performance against a test set, providing metrics such as accuracy, recall, precision, and F1-score.

---

## How to Run the Scripts
Below are the step-by-step instructions to run the pipeline for the ScAN Suicide Analysis Project. Make sure to execute the scripts from the project's root directory unless otherwise specified.

### Preprocessing

1. **Extract EHR Notes**: Use `get_HADM_files.py` script to retrieve and structure Electronic Health Records (EHR) from the MIMIC-III dataset. This code is based on work by the original author available
	here: https://github.com/bsinghpratap/ScAN/tree/main/get_data/scripts

#### Steps to Run:

Execute the script with:
   ```bash
   python scripts/get_HADM_files.py
   ```

	**Output**: EHR notes will be systematically organized and stored in the data/mimic3_notes/ directory. Please ensure that the required MIMIC-III files are correctly placed in the expected directories before running this script.

2. **Prepare SI/SA Annotations**: Preprocess and synchronize the ScAN annotations with the clinical notes to ready the data for modeling. The `Preprocess-Annotations-for-ScAN-predictor.py` script standardizes the clinical notes format and associates them with the corresponding annotations.

Execute the script with:

   Navigate to the `scripts/` directory:
   ```bash
   cd scripts
   python Preprocess-Annotations-for-ScAN-predictor.py
   ```
   
   **Output**: Preprocessed data will be saved as `preprocessed_annotations.csv`. Any issues or logs will be recorded in preprocessing.log.
   **Purpose**:This step ensures that the raw clinical notes are appropriately structured and annotated for effective training and evaluation in the project pipeline.   
   
3. **Train and Evaluate Prediction Model**: The `Prediction_module_model.py script` is used to train a neural network to classify instances of suicidal ideation and suicide attempts from the preprocessed clinical text.

#### Steps to Run:
   Navigate to the `scripts/` directory:
   ```bash
   cd scripts
   python scripts/Prediction_module_model.py
   ```
   
   **Output**: The script will output the model's performance metrics and include details such as accuracy, precision, recall, and F1 scores. Additionally, it generates files with prediction results. 
   **Purpose**: This script leverages the power of RoBERTa for NLP tasks, focusing on classifying and evaluating high-risk profiles within the clinical notes data.
   
## Installation Guide

Follow the steps below to set up your environment for running the code in this repository.

### Step 1: Clone the Repository

Clone the repository to your local machine:
```bash
git clone https://github.com/icbautista/HighRiskProject.git
cd HighRiskProject
```
### Step 2: Set Up a Virtual Environment (Optional)
- It is recommended to use a virtual environment to avoid conflicts with existing Python packages. You can use venv or conda for this purpose.
Using venv:
```bash
python3 -m venv highrisk-env
source highrisk-env/bin/activate  # On Windows: highrisk-env\Scripts\activate 
```
Using conda:
```bash
conda create -n highrisk-env python=3.8 -y
conda activate highrisk-env
```

### Step 3: Install Dependencies
- Install the required Python packages listed in the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

- **Datasets**
Ensure you have access to the **MIMIC-III** and **ScAN** datasets. Place the them in the directory as required by the scripts.
Once installed, proceed to the How to Run the Code section to execute the pipeline.
