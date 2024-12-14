"""
FILE: Preprocess-Annotations-for-ScAN-predictor.py
AUTHOR: Igor Carreon
DESCRIPTION:
This script preprocesses annotated data for the Suicide Attempt and Ideation (SA/SI)
prediction with the ScAN predictor. It reads clinical notes and the associated 
annotations for SUICIDE_ATTEMPT and SUICIDE_IDEATION, processes the text data 
to identify relevant text spans, and labels them as positive or negative evidence 
based on predefined SA/SI categories. The script tokenizes the text, adds noise to 
the dataset to improve robustness, and finally writes the preprocessed data to a CSV file.

INPUTS:
- mimic3_notes/: Directory containing full clinical notes (EHR texts).
- train_hadm.json: JSON containing annotations that label parts of clinical notes
  relevant to suicide attempt or ideation.

OUTPUTS:
- preprocessed_annotations.csv: CSV containing annotations, their span, and tokens after
  preprocessing, along with evidence classification.
"""
import json
import pandas as pd
import os
import random
import logging
from nltk import sent_tokenize
from clinical_sectionizer import TextSectionizer
from transformers import RobertaTokenizer

# Configure logging to file with INFO level messages.
logging.basicConfig(filename='preprocessing_val.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# File paths and directories
ehr_text_dir = 'mimic3_notes/' # Directory containing EHR notes
annotations_file = 'train_hadm.json'  # Input JSON file
output_csv = 'preprocessed_annotations.csv'  # Output CSV file

# Initialize the TextSectionizer and tokenizer from the transformers library
sectionizer = TextSectionizer()
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def add_noisy_paragraphs(evidence_paragraphs, noise_ratio=0.05):
    """
    Add irrelevant paragraphs as noise to the evidence paragraphs.
    """
    noisy_paragraphs = [
        {
            'text_span': "This is an irrelevant paragraph.",
            'evidence': 'evidence_no',
            'SA_CATEGORY': 'Neutral-SA',
            'SI_CATEGORY': 'Neutral-SI',
            'tokens': []
        }
        for _ in range(int(len(evidence_paragraphs) * noise_ratio))
    ]
    return evidence_paragraphs + noisy_paragraphs

def label_paragraph(paragraph, sa_category, si_category):
    """
    Label the paragraph as evidence based on SA/SI categories.
    """
    if sa_category == 'Neutral-SA' and si_category == 'Neutral-SI':
        return 'evidence_no'
    else: 
        return 'evidence_yes'

def determine_category(details):
    """
    Determine SA and SI categories based on details.
    """
    sa_category, si_category = 'Neutral-SA', 'Neutral-SI'
    if details:
        category = details.get('category', 'Unknown')
        status = details.get('status', 'Unknown')
        if 'suicide_attempt' in details:
            if category in ['T36-T50', 'T14.91', 'T51-T65', 'X71-X83', 'T71']:
                sa_category = 'SA_Positive'
            elif category == 'unsure':
                sa_category = 'SA_Unsure'
            elif category == 'N/A':
                sa_category = 'SA_Negative'
        elif 'suicide_ideation' in details:
            if status == 'present':
                si_category = 'SI_Positive'
            elif status == 'absent':
                si_category = 'SI_Negative'
    return sa_category, si_category

def tokenize_text(text):
    """
    Tokenize the input text using the tokenizer.
    """
    if not text.strip():
        return []
    try:
        return tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=512)
    except Exception as e:
        logging.error(f"Error tokenizing text: {text[:30]}... - {e}")
        return []

# Load JSON data
with open(annotations_file, 'r') as file:
    data = json.load(file)

annotations = []

# Process each hospital admission
for hadm_id, instances in data.items():
    subject_id, hadm_id = hadm_id.split('_')
    ehr_text_path = os.path.join(ehr_text_dir, f'{subject_id}_{hadm_id}')
    
    # Load the EHR note
    ehr_text = ""
    if os.path.exists(ehr_text_path):
        with open(ehr_text_path, 'r', encoding='utf-8') as f:
            ehr_text = f.read()
    else:
        logging.warning(f'EHR text file not found for SUBJECT_ID: {subject_id}, HADM_ID: {hadm_id}')
    
    # Sectionize the EHR text if not empty
    sections = sectionizer(ehr_text) if ehr_text else []
    
    # If no instances exist
    if not instances:
        sa_category, si_category = determine_category(None)
        annotations.append({
            'subject_id': subject_id,
            'hadm_id': hadm_id,
            'instance_id': 'None',
            'start_idx': 'None',
            'end_idx': 'None',
            'text_span': 'None',
            'category': 'Unknown',
            'period': 'Unknown',
            'frequency': 'Unknown',
            'suicide_attempt': 'Unknown',
            'SA_CATEGORY': sa_category,
            'SI_CATEGORY': si_category,
            'tokens': [],
            'evidence': label_paragraph('None', sa_category, si_category)  # Use label_paragraph here
        })
        continue
    
    # Process each instance
    for instance_id, instance_data in instances.items():
        annotation_info = instance_data.get('annotation', [])
        category = instance_data.get('category', 'Unknown')
        period = instance_data.get('period', 'Unknown')
        frequency = instance_data.get('frequency', 'Unknown')
        
        text_span = ""
        start_idx = None
        end_idx = None
        
        if annotation_info and len(annotation_info) >= 2 and annotation_info[0] and annotation_info[1]:
            start_idx = int(annotation_info[0])
            end_idx = int(annotation_info[1])
            
            for section_title, section_header, section_text in sections:
                section_start = ehr_text.find(section_text)
                section_end = section_start + len(section_text)

                if section_start <= start_idx < section_end:
                    context_start_idx = max(section_start, start_idx)
                    context_end_idx = min(section_end, end_idx)
                    text_span = ehr_text[context_start_idx:context_end_idx]
                    break
        else:
            text_span = 'None'
        
        # Determine SA and SI categories
        sa_category, si_category = determine_category(instance_data)
        
        # Ensure mutual exclusivity
        if sa_category != 'Neutral-SA':
            si_category = 'Neutral-SI'
        elif si_category != 'Neutral-SI':
            sa_category = 'Neutral-SA'
        
        # Tokenize text span
        tokens = tokenize_text(text_span) if text_span != 'None' else []
        
        # Append the annotation
        annotations.append({
            'subject_id': subject_id,
            'hadm_id': hadm_id,
            'instance_id': instance_id,
            'start_idx': start_idx if start_idx is not None else 'None',
            'end_idx': end_idx if end_idx is not None else 'None',
            'text_span': text_span if text_span else 'None',
            'category': category,
            'period': period,
            'frequency': frequency,
            'suicide_attempt': instance_data.get('suicide_attempt', 'Unknown'),
            'SA_CATEGORY': sa_category,
            'SI_CATEGORY': si_category,
            'tokens': tokens,
            'evidence': label_paragraph(text_span, sa_category, si_category)  # Use label_paragraph here
        })

df = pd.DataFrame(annotations)

# Create DataFrame from annotations and add noise with define functions.
df_records = df.to_dict('records')  
df_with_noise = add_noisy_paragraphs(df_records)  # Add noisy paragraphs
df = pd.DataFrame(df_with_noise) 

# Save the DataFrame to CSV and log completion
df.to_csv(output_csv, index=False)
logging.info(f"Annotations saved to {output_csv}.")
