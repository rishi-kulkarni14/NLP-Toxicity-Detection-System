import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModel,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
import wandb
import os
from typing import List, Dict, Tuple, Optional
import json
import gc

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 2e-5
MODEL_NAME = "roberta-base"
RANDOM_SEED = 42

# Set random seeds
def set_seed(seed_value: int):
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

class ToxicityDataset(Dataset):
    """Custom dataset for toxicity classification"""
    
    def __init__(self, texts: List[str], labels: Optional[np.ndarray], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
        if self.labels is not None:
            item['labels'] = torch.FloatTensor(self.labels[idx])
            
        return item

class ToxicityClassifier(nn.Module):
    """Multi-label toxicity classifier using transformer architecture"""
    
    def __init__(self, n_labels: int, dropout: float = 0.3):
        super(ToxicityClassifier, self).__init__()
        self.roberta = AutoModel.from_pretrained(MODEL_NAME)
        
        # Classifier head
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(self.roberta.config.hidden_size, 512)
        self.dropout2 = nn.Dropout(dropout)
        self.output = nn.Linear(512, n_labels)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(pooled_output)
        x = self.dense(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        return self.sigmoid(self.output(x))

class ToxicityDetector:
    """Main class for toxicity detection system"""
    
    def __init__(self, model_name: str = MODEL_NAME):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label_columns = [
            'toxic',
            'severe_toxic',
            'obscene',
            'threat',
            'insult',
            'identity_hate'
        ]
        self.model = ToxicityClassifier(len(self.label_columns)).to(self.device)
        
        # Initialize wandb if environment variable is set
        try:
            if os.getenv('USE_WANDB'):
                wandb.init(project="toxicity-detector", mode="online")
            else:
                wandb.init(project="toxicity-detector", mode="disabled")
        except:
            logger.warning("Wandb initialization failed, continuing without logging")
        
        logger.info(f"Using device: {self.device}")

    def prepare_data(self, data_path: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare data loaders from CSV file"""
        logger.info("Loading and preparing data...")
        
        # Load data
        df = pd.read_csv(data_path)
        
        # Split data
        train_df, temp_df = train_test_split(
            df,
            test_size=0.3,
            random_state=RANDOM_SEED,
            stratify=df['toxic']
        )
        
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=RANDOM_SEED,
            stratify=temp_df['toxic']
        )
        
        # Create datasets
        train_dataset = ToxicityDataset(
            texts=train_df['comment_text'].values,
            labels=train_df[self.label_columns].values,
            tokenizer=self.tokenizer,
            max_length=MAX_LENGTH
        )
        
        val_dataset = ToxicityDataset(
            texts=val_df['comment_text'].values,
            labels=val_df[self.label_columns].values,
            tokenizer=self.tokenizer,
            max_length=MAX_LENGTH
        )
        
        test_dataset = ToxicityDataset(
            texts=test_df['comment_text'].values,
            labels=test_df[self.label_columns].values,
            tokenizer=self.tokenizer,
            max_length=MAX_LENGTH
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            num_workers=2
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            num_workers=2
        )
        
        return train_loader, val_loader, test_loader

    def train_epoch(
        self,
        data_loader: DataLoader,
        optimizer,
        scheduler,
        epoch: int
    ) -> Tuple[float, np.ndarray]:
        """Train for one epoch"""
        self.model.train()
        losses = []
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(data_loader, desc=f'Epoch {epoch + 1}')
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            optimizer.zero_grad()
            
            outputs = self.model(input_ids, attention_mask)
            loss = nn.BCELoss()(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            losses.append(loss.item())
            all_predictions.extend(outputs.cpu().detach().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({'loss': f'{loss.item():.3f}'})
        
        epoch_loss = np.mean(losses)
        epoch_auc = roc_auc_score(
            np.array(all_labels),
            np.array(all_predictions),
            average='weighted'
        )
        
        return epoch_loss, epoch_auc

    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Evaluate the model"""
        self.model.eval()
        losses = []
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Evaluating'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = nn.BCELoss()(outputs, labels)
                
                losses.append(loss.item())
                all_predictions.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = np.mean(losses)
        auc_score = roc_auc_score(
            np.array(all_labels),
            np.array(all_predictions),
            average='weighted'
        )
        
        return (
            avg_loss,
            auc_score,
            np.array(all_predictions),
            np.array(all_labels)
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader
    ):
        """Complete training pipeline"""
        optimizer = AdamW(self.model.parameters(), lr=LEARNING_RATE)
        total_steps = len(train_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        best_auc = 0
        
        for epoch in range(EPOCHS):
            logger.info(f'\nEpoch {epoch + 1}/{EPOCHS}')
            
            train_loss, train_auc = self.train_epoch(
                train_loader,
                optimizer,
                scheduler,
                epoch
            )
            
            val_loss, val_auc, val_preds, val_labels = self.evaluate(val_loader)
            
            # Log metrics if wandb is enabled
            try:
                wandb.log({
                'train_loss': train_loss,
                'train_auc': train_auc,
                'val_loss': val_loss,
                'val_auc': val_auc,
                'epoch': epoch + 1
            })
            except:
                pass
            
            logger.info(f'Train Loss: {train_loss:.3f}, AUC: {train_auc:.3f}')
            logger.info(f'Val Loss: {val_loss:.3f}, AUC: {val_auc:.3f}')
            
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(self.model.state_dict(), 'best_model.bin')
                self.plot_roc_curves(val_labels, val_preds)
        
        # Final evaluation on test set
        logger.info('\nEvaluating on test set...')
        test_loss, test_auc, test_preds, test_labels = self.evaluate(test_loader)
        logger.info(f'Test Loss: {test_loss:.3f}, AUC: {test_auc:.3f}')
        
        # Generate detailed classification report
        self.generate_classification_report(test_labels, test_preds > 0.5)
        
        wandb.finish()

    def plot_roc_curves(self, labels: np.ndarray, predictions: np.ndarray):
        """Plot ROC curves for each toxicity type"""
        plt.figure(figsize=(12, 8))
        
        for i, label in enumerate(self.label_columns):
            fpr, tpr, _ = roc_curve(labels[:, i], predictions[:, i])
            auc = roc_auc_score(labels[:, i], predictions[:, i])
            plt.plot(fpr, tpr, label=f'{label} (AUC = {auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves by Toxicity Type')
        plt.legend()
        plt.tight_layout()
        
        plt.savefig('roc_curves.png')
        wandb.log({"roc_curves": wandb.Image('roc_curves.png')})
        plt.close()

    def generate_classification_report(self, labels: np.ndarray, predictions: np.ndarray):
        """Generate and save detailed classification report"""
        report = {}
        
        for i, label in enumerate(self.label_columns):
            label_report = classification_report(
                labels[:, i],
                predictions[:, i],
                output_dict=True
            )
            report[label] = label_report
        
        # Save report
        with open('classification_report.json', 'w') as f:
            json.dump(report, f, indent=4)
        
        # Log to wandb
        wandb.save('classification_report.json')

    def predict(self, texts: List[str]) -> Dict[str, List[float]]:
        """Predict toxicity probabilities for new texts"""
        self.model.eval()
        
        dataset = ToxicityDataset(
            texts=texts,
            labels=None,
            tokenizer=self.tokenizer,
            max_length=MAX_LENGTH
        )
        
        loader = DataLoader(dataset, batch_size=BATCH_SIZE)
        predictions = []
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                predictions.extend(outputs.cpu().numpy())
        
        # Convert predictions to dictionary
        results = {
            text: {
                label: float(pred[i])
                for i, label in enumerate(self.label_columns)
            }
            for text, pred in zip(texts, predictions)
        }
        
        return results




import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def main():
    """Main execution function"""
    # Set random seed
    set_seed(RANDOM_SEED)
    
    # Initialize detector
    detector = ToxicityDetector()
    
    # Prepare data
    train_loader, val_loader, test_loader = detector.prepare_data('toxic_comments.csv')
    
    # Train model
    detector.train(train_loader, val_loader, test_loader)
    
    # Example predictions
    sample_texts = [
        "This is a wonderful contribution to the discussion!",
        "I completely disagree with your opinion, but respect your perspective.",
        "You're all idiots and should be banned.",
        "Great point, thanks for sharing!"
    ]
    
    predictions = detector.predict(sample_texts)
    
    print("\nSample Predictions:")
    for text, scores in predictions.items():
            print(f"{label}: {score:.3f}")


class ToxicityDetector:
   def __init__(self):
       # Use a model specifically fine-tuned for toxicity
       model_name = "unitary/toxic-bert"
       self.tokenizer = AutoTokenizer.from_pretrained(model_name)
       self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
       self.labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
       self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       self.model.to(self.device)

   def predict(self, texts):
       predictions = {}
       for text in texts:
           # Tokenize text
           inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
           inputs = {k: v.to(self.device) for k, v in inputs.items()}
           
           # Get predictions
           with torch.no_grad():
               outputs = self.model(**inputs)
               probabilities = torch.sigmoid(outputs.logits)[0]
           
           # Convert to dictionary
           scores = {label: float(prob) for label, prob in zip(self.labels, probabilities)}
           predictions[text] = scores
           
       return predictions

if __name__ == "__main__":
   # Initialize detector
   detector = ToxicityDetector()
   
   while True:
       # Get user input
       print("\nEnter a text to analyze (or 'quit' to exit):")
       user_text = input().strip()
       
       if user_text.lower() == 'quit':
           print("Thank you for using the toxicity detector!")
           break
           
       if not user_text:
           print("Please enter some text to analyze.")
           continue
       
       # Get predictions
       predictions = detector.predict([user_text])
       
       # Display results
       print("\nAnalysis Results:")
       print("-" * 50)
       for text, scores in predictions.items():
           sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
           for label, score in sorted_scores.items():
               warning = "⚠️ " if score > 0.5 else "   "
               print(f"{warning}{label}: {score:.3f}")