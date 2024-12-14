"""
FILE: Prediction_module_model.py
AUTHOR: Igor Carreon
DESCRIPTION:
This script serves as a suicide risk classification module that loads embeddings of clinical
text and trains a model to predict Suicide Attempt (SA) and Suicide Ideation (SI) risk levels.
It utilizes pretrained RoBERTa embeddings and a neural network classification model. The
classification tasks rely on attention mechanisms and linear layers tailored to the specific
number of SA and SI categories. The model's performance is evaluated using confusion matrices
and detailed classification reports, and the script handles data preparation, model training,
and evaluation in a single integrated workflow.

INPUTS:
- mimic3_notes/: Directory containing full clinical notes (presumably preprocessed for embedding extraction).
- preprocessed_annotations.csv: CSV file containing preprocessed annotations that align with clinical text spans.
- all_matched_embeddings.pt: PyTorch tensor file containing matched paragraph embeddings if previously saved.
- all_matched_labels.pt: PyTorch tensor file containing matched paragraph labels if previously saved.

OUTPUTS:
- trained_sa_si_model: The trained neural network model for SA/SI prediction (if saved within the script).
- evidence_processing.log: Log file for detailed processing information.
- Various prints and plots for manual inspection, such as confusion matrices visualized with Seaborn.
"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.utils.rnn as rnn_utils
import logging
from datetime import datetime
import torch.nn as nn
from nltk.tokenize import sent_tokenize
import ast
import torch
from torch.utils.data import Dataset, DataLoader

# Setting CUDA_LAUNCH_BLOCKING for better error traceback during CUDA operations
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Define model save path and data file paths
script_dir = os.path.dirname(__file__)
model_save_path = os.path.join(script_dir, "trained_sa_si_model")
embeddings_file = os.path.join(script_dir, "all_matched_embeddings.pt")
labels_file = os.path.join(script_dir, "all_matched_labels.pt")

# Configure logging for debugging and information tracking
logging.basicConfig(
    filename='evidence_processing.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants for label mapping
LABEL_MAPPING = {
    'SA_Positive': 0,
    'SA_Unsure': 1,
    'SA_Negative': 2,
    'Neutral-SA': 3,
    'SI_Positive': 4,
    'SI_Negative': 5,
    'Neutral-SI': 6
}

# Define the directory containing clinical notes
notes_directory = "./mimic3_notes"

# Custom PyTorch module for SA/SI classification
class SA_SI_PredictionModule(nn.Module):
    def __init__(self, retriever_model, num_attention_heads=3, sa_classes=4, si_classes=4):
        """
        A module for predicting SA (Suicidal Attempt) and SI (Suicidal Ideation) categories.

        Parameters:
            retriever_model (nn.Module): A pre-trained transformer model (e.g., RobertaModel) used for embeddings.
            num_attention_heads (int): Number of attention heads for the MultiheadAttention layer.
            sa_classes (int): Number of SA classification categories.
            si_classes (int): Number of SI classification categories.
        """
        super(SA_SI_PredictionModule, self).__init__()
        self.retriever = retriever_model
        self.attention = nn.MultiheadAttention(embed_dim=768, num_heads=num_attention_heads, batch_first=True)
        self.sa_classifier = nn.Linear(768, sa_classes)
        self.si_classifier = nn.Linear(768, si_classes)
        self.v0 = nn.Parameter(torch.randn(1, 1, 768))  # Learnable vector for initial prediction

    def forward(self, paragraph_embeddings):
        """
        Forward pass for SA and SI prediction.

        Parameters:
            paragraph_embeddings (torch.Tensor): Input embeddings of shape (batch_size, seq_length, embed_dim).

        Returns:
            sa_logits (torch.Tensor): Logits for SA classification.
            si_logits (torch.Tensor): Logits for SI classification.
        """
        if paragraph_embeddings.dim() == 4 and paragraph_embeddings.size(2) == 1:
            paragraph_embeddings = paragraph_embeddings.squeeze(2)  # Remove singleton dimension

        batch_size = paragraph_embeddings.size(0)
        v0 = self.v0.expand(batch_size, 1, 768)  # Expand v0 to batch size
        embeddings = torch.cat([v0, paragraph_embeddings], dim=1)
        attention_output, _ = self.attention(embeddings, embeddings, embeddings)
        h0 = attention_output[:, 0, :]  # Use the first vector for classification
        sa_logits = self.sa_classifier(h0)
        si_logits = self.si_classifier(h0)
        return sa_logits, si_logits
    
def match_paragraphs_to_evidence(paragraphs_data, annotations_df, subject_id, hadm_id, tokenizer, device, retriever_model):
    
    matched_embeddings = []
    matched_labels = []

    print(f"Processing note for subject_id: {subject_id}, hadm_id: {hadm_id}")
    print(f"=====================================================================")
    # Filter the annotations to the current subject_id and hadm_id
    evidence_annotations = annotations_df[
        (annotations_df['subject_id'] == subject_id) &
        (annotations_df['hadm_id'] == hadm_id)
    ]

    print(f"Number of paragraphs being processed: {len(paragraphs_data)}")
    print(f"Number of evidence sentences from annotations: {len(evidence_annotations)}")

    # Iterate over the paragraphs data and match them to the evidence annotations
    for i, paragraph_data in enumerate(paragraphs_data):
        paragraph = paragraph_data['paragraph']
        # Tokenize the entire paragraph once
        paragraph_tokens = tokenizer.encode(paragraph, add_special_tokens=True)
        paragraph_token_set = set(paragraph_tokens)

        # Check for overlap with any evidence sentence tokens
        for j, evidence_row in evidence_annotations.iterrows():
            evidence_tokens = evidence_row['tokens']
            if isinstance(evidence_tokens, str):
                evidence_tokens = ast.literal_eval(evidence_tokens)
            
            # Check the length of the evidence tokens list
            if len(evidence_tokens) > 5:                   
                evidence_token_set = set(evidence_tokens[3:])
            else:
                evidence_token_set = set(evidence_tokens)

            # Compare tokens to find a match, ignoring special tokens
            if evidence_token_set.issubset(paragraph_token_set):
                #print(f"ATTENTION: Match found for evidence {j} in paragraph {i}:")
                embeddings = extract_paragraph_embeddings(paragraph, tokenizer, retriever_model, device)
                matched_embeddings.append(embeddings)
                sa_label = LABEL_MAPPING[evidence_row['SA_CATEGORY']]
                si_label = LABEL_MAPPING[evidence_row['SI_CATEGORY']]
                matched_labels.append((sa_label, si_label))
                print(f"ANNOTATED Evidence Sentence: '{evidence_row['text_span']}'")
                print(f"ANNOTATED Evidence Instance: '{evidence_row['instance_id']}'")
                print(f"SA_CATEGORY - {evidence_row['SA_CATEGORY']}, SI_CATEGORY - {evidence_row['SI_CATEGORY']}")
                print ("PARAGRAPH: ", paragraph)
                print ('----------------------------------------------------------------------------------------------------')
        #else:
            #print(f"No match found for paragraph {i}: {paragraph}")

    return matched_embeddings, matched_labels

# Define the function to extract embeddings
def extract_paragraph_embeddings(paragraph, tokenizer, model, device):
    # Tokenize the paragraph and move tokens to the correct device
    tokens = tokenizer(paragraph, padding=True, truncation=True, return_tensors="pt", max_length=512)
    tokens = {key: value.to(device) for key, value in tokens.items()}
        
    with torch.no_grad():
        # Pass tokens through the model and calculate the mean of the last hidden state
        embeddings = model(**tokens).last_hidden_state.mean(dim=1)
    
    # The embeddings are already on the correct device, so we can return them directly
    return embeddings

def process_note_embeddings(note_file, tokenizer, retriever_model, device):
    file_path = os.path.join(notes_directory, note_file)
    try:
        with open(file_path, 'r') as file:
            note_text = file.read()
    except FileNotFoundError:
        logging.error(f"The file {file_path} does not exist.")
        return []

    # Split the note text into paragraphs
    paragraphs = note_text.split('\n\n')  # Simple paragraph tokenization based on double newlines
    paragraph_embeddings = []

    for i, paragraph in enumerate(paragraphs):
        if isinstance(paragraph, str) and paragraph.strip():  # Check if paragraph is a string and not empty
            embeddings = extract_paragraph_embeddings(paragraph, tokenizer, retriever_model, device)
            paragraph_embeddings.append({'paragraph': paragraph, 'embedding': embeddings})
        else:
            logging.error(f"Paragraph {i} is not a string or is empty. Actual content: {repr(paragraph)}")

    return paragraph_embeddings

def evaluate_model(sa_logits, si_logits, true_labels):
    sa_probs = torch.softmax(sa_logits, dim=-1)
    si_probs = torch.softmax(si_logits, dim=-1)

    sa_predictions = torch.argmax(sa_probs, dim=-1).cpu().numpy()
    si_predictions = torch.argmax(si_probs, dim=-1).cpu().numpy()

    # Define SA and SI-related labels
    sa_labels = {k: v for k, v in LABEL_MAPPING.items() if "SA" in k}
    si_labels = {k: v for k, v in LABEL_MAPPING.items() if "SI" in k}

    # Filter valid labels for SA and SI
    valid_sa_labels = [label for label in sa_labels.values() if label in true_labels["sa"]]
    valid_si_labels = [label for label in si_labels.values() if label in true_labels["si"]]

    # Check if there are valid labels before computing the confusion matrix
    if valid_sa_labels:
        sa_cm = confusion_matrix(true_labels["sa"], sa_predictions, labels=valid_sa_labels)
        print("SA Confusion Matrix:")
        print(sa_cm)
    else:
        print("No valid SA labels found for confusion matrix.")

    if valid_si_labels:
        si_cm = confusion_matrix(true_labels["si"], si_predictions, labels=valid_si_labels)
        print("SI Confusion Matrix:")
        print(si_cm)
    else:
        print("No valid SI labels found for confusion matrix.")

    print("SA Classification Report:")
    print(classification_report(
        true_labels["sa"],
        sa_predictions,
        target_names=list(sa_labels.keys()), 
        labels=list(sa_labels.values())  # Explicitly specify expected SA labels
    ))

    print("SI Classification Report:")
    print(classification_report(
        true_labels["si"],
        si_predictions,
        target_names=list(si_labels.keys()),
        labels=list(si_labels.values())  # Explicitly specify expected SI labels
    ))

    # Debugging: Filter valid labels for SA and SI
    valid_sa_labels = [label for label in sa_labels.values() if label in true_labels["sa"]]
    valid_si_labels = [label for label in si_labels.values() if label in true_labels["si"]]

    # Compute confusion matrix only for valid labels
    sa_cm = confusion_matrix(
        true_labels["sa"], 
        sa_predictions, 
        labels=valid_sa_labels
    )
    si_cm = confusion_matrix(
        true_labels["si"], 
        si_predictions, 
        labels=valid_si_labels
    )

    # Debugging: Print the confusion matrix
    print("SA Confusion Matrix:")
    print(sa_cm)
    print("SI Confusion Matrix:")
    print(si_cm)

    # Visualize confusion matrices
    plt.figure(figsize=(10, 8))
    sns.heatmap(sa_cm, annot=True, fmt="d", cmap="Blues", xticklabels=sa_labels.keys(), yticklabels=sa_labels.keys())
    plt.title("SA Confusion Matrix")
    plt.show(block=False)

    plt.figure(figsize=(10, 8))
    sns.heatmap(si_cm, annot=True, fmt="d", cmap="Blues", xticklabels=si_labels.keys(), yticklabels=si_labels.keys())
    plt.title("SI Confusion Matrix")
    plt.show(block=False)

class SuicideRiskDataset(Dataset):
    def __init__(self, embeddings, sa_labels, si_labels):
        """
        Custom dataset for suicide risk prediction.

        Parameters:
        embeddings (list of torch.Tensor): List of paragraph embeddings.
        sa_labels (list of int): List of SA labels for each paragraph.
        si_labels (list of int): List of SI labels for each paragraph.
        """
        self.embeddings = embeddings
        self.sa_labels = sa_labels
        self.si_labels = si_labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {
            "embeddings": self.embeddings[idx],
            "sa_labels": self.sa_labels[idx],
            "si_labels": self.si_labels[idx],
        }

# The main workflow for model training and evaluation
def main():
    # Record the start time
    start_time = datetime.now()
    print(f"Process started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Set up the device for PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    retriever_model = RobertaModel.from_pretrained("roberta-base").to(device)
    prediction_model = SA_SI_PredictionModule(retriever_model).to(device)

    # Load the preprocessed annotations file and specify dtype for subject_id and hadm_id
    annotations_df = pd.read_csv("./preprocessed_annotations.csv", dtype={'subject_id': str, 'hadm_id': str})

    # Check if the processed data already exists
    if os.path.exists(embeddings_file) and os.path.exists(labels_file):
        # Load preprocessed embeddings and labels
        all_matched_embeddings = torch.load(embeddings_file)
        all_matched_labels = torch.load(labels_file)
    else:        
        # Initialize lists to collect embeddings and labels from all files
        all_matched_embeddings = []
        all_matched_labels = []

        # Process each note and extract embeddings
        for note_file in os.listdir(notes_directory):
            # Extract subject_id and hadm_id from the filename
            subject_id, hadm_id = note_file.split('_')[:2]

            # Ensure the parsed IDs are strings
            subject_id = str(subject_id)
            hadm_id = str(hadm_id)

            # Process the note and extract embeddings
            paragraphs_data = process_note_embeddings(note_file, tokenizer, retriever_model, device)

            # Match the paragraph embeddings to the annotated evidence
            matched_embeddings, matched_labels = match_paragraphs_to_evidence(
                paragraphs_data, annotations_df, subject_id, hadm_id, tokenizer, device, retriever_model
            )    

            # Append the results from this file to the aggregated lists
            all_matched_embeddings.extend(matched_embeddings)
            all_matched_labels.extend(matched_labels)

        # Convert embeddings to a tensor
        all_matched_embeddings = torch.stack(all_matched_embeddings)

        # Save processed embeddings and labels to disk
        torch.save(all_matched_embeddings, embeddings_file)
        torch.save(all_matched_labels, labels_file)
    
    # Convert labels to tensors and split SA and SI labels
    sa_labels = torch.tensor([label[0] for label in all_matched_labels], dtype=torch.long)
    si_labels = torch.tensor([label[1] for label in all_matched_labels], dtype=torch.long)

    # Check the range of labels and the number of classes defined in the model
    print("SA label range:", min(sa_labels).item(), "to", max(sa_labels).item())
    print("SI label range:", min(si_labels).item(), "to", max(si_labels).item())

    # Assuming prediction_model is already defined and loaded
    print("Number of SA classes in the model:", prediction_model.sa_classifier.out_features)
    print("Number of SI classes in the model:", prediction_model.si_classifier.out_features)

    # Remap SI labels to start from 0
    si_labels -= si_labels.min()

    # Now check the label ranges again
    print("Adjusted SI label range:", min(si_labels).item(), "to", max(si_labels).item())

    # Ensure that the number of classes in the model matches the range of labels
    assert max(sa_labels).item() < prediction_model.sa_classifier.out_features
    assert max(si_labels).item() < prediction_model.si_classifier.out_features

    # Create the dataset
    suicide_risk_dataset = SuicideRiskDataset(all_matched_embeddings, sa_labels, si_labels)

    # Create a DataLoader
    batch_size = 16  # Batch size
    suicide_risk_dataloader = DataLoader(suicide_risk_dataset, batch_size=batch_size, shuffle=True)

    # Define loss functions for SA and SI classification tasks
    sa_loss_fn = torch.nn.CrossEntropyLoss()
    si_loss_fn = torch.nn.CrossEntropyLoss()

    # Define an optimizer (e.g., Adam)
    optimizer = torch.optim.Adam(prediction_model.parameters(), lr=1e-4)

    # Collect true labels for evaluation
    true_sa_labels = [label[0] for label in all_matched_labels]
    true_si_labels = [label[1] for label in all_matched_labels]

    num_epochs = 1  # Define the number of epochs

    # Training loop
    for epoch in range(num_epochs):
        prediction_model.train()  # Set the model to training mode
        for batch in suicide_risk_dataloader:
            embeddings = batch["embeddings"].to(device)
            sa_labels_batch = batch["sa_labels"].to(device)
            si_labels_batch = batch["si_labels"].to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            sa_logits, si_logits = prediction_model(embeddings)

            # Calculate loss for both SA and SI
            sa_loss = sa_loss_fn(sa_logits, sa_labels_batch)
            si_loss = si_loss_fn(si_logits, si_labels_batch)
            total_loss = sa_loss + si_loss  # Combine losses 

            # Backward pass
            total_loss.backward()

            # Update model parameters
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss.item()}")

    # Set the model to evaluation mode
    prediction_model.eval()

    # Evaluation loop (after training)
    sa_logits_all, si_logits_all = [], []
    with torch.no_grad():  # No gradient is needed for evaluation
        for batch in suicide_risk_dataloader:
            embeddings = batch["embeddings"].to(device)
            sa_logits, si_logits = prediction_model(embeddings)
            sa_logits_all.append(sa_logits.cpu())
            si_logits_all.append(si_logits.cpu())

    # Concatenate logits from all batches
    sa_logits_all = torch.cat(sa_logits_all, dim=0)
    si_logits_all = torch.cat(si_logits_all, dim=0)

    # Convert true labels to a tensor for evaluation
    true_sa_labels = torch.tensor(true_sa_labels, dtype=torch.long)
    true_si_labels = torch.tensor(true_si_labels, dtype=torch.long)

    # Evaluate the model
    evaluate_model(sa_logits_all, si_logits_all, {"sa": true_sa_labels, "si": true_si_labels})

    # Record the end time
    end_time = datetime.now()
    print(f"Process ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Calculate the duration
    duration = end_time - start_time
    print(f"Total processing time: {duration}")
