from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForMaskedLM
import tkinter as tk
from tkinter import scrolledtext
from tqdm import tqdm
import gc

class TextDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=128):
        self.encodings = tokenizer(
            dataset['text'],
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
    
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}
    
    def __len__(self):
        return len(self.encodings.input_ids)

class EnhancedBERTAutocomplete(nn.Module):
    def __init__(self):
        super(EnhancedBERTAutocomplete, self).__init__()
        print("Loading pre-trained BERT model...")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.model.eval()
        
    def get_suggestions(self, text):
        # Get last N words for better context
        words = text.strip().split()
        context = ' '.join(words[-5:] if len(words) > 5 else words)
        
        # Create multiple masked versions for better prediction
        masked_texts = [
            context + " " + self.tokenizer.mask_token,
            context + " " + self.tokenizer.mask_token + ".",
            context + " " + self.tokenizer.mask_token + " is",
        ]
        
        all_suggestions = []
        for masked_text in masked_texts:
            inputs = self.tokenizer(
                masked_text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                mask_position = (inputs.input_ids == self.tokenizer.mask_token_id)[0].nonzero().item()
                logits = outputs.logits[0, mask_position]
                
                # Apply temperature scaling
                temperature = 0.7
                scaled_logits = logits / temperature
                
                # Get probabilities
                probs = torch.softmax(scaled_logits, dim=-1)
                confidence_threshold = 0.1
                
                # Get top predictions with confidence threshold
                top_k = torch.topk(probs, k=10)
                
                for score, token_id in zip(top_k.values, top_k.indices):
                    if score > confidence_threshold:
                        word = self.tokenizer.decode([token_id])
                        if (word.strip() and 
                            len(word) > 2 and
                            not word.startswith('[') and
                            not word.startswith('#') and
                            word.isalpha()):
                            all_suggestions.append((word, score.item()))
        
        # Sort by confidence and get unique suggestions
        all_suggestions.sort(key=lambda x: x[1], reverse=True)
        unique_suggestions = []
        seen = set()
        for word, _ in all_suggestions:
            if word not in seen:
                seen.add(word)
                unique_suggestions.append(word)
                if len(unique_suggestions) >= 5:
                    break
                    
        return unique_suggestions

class TextAutocompleteUI:
    def __init__(self, model):
        self.model = model
        self.window = tk.Tk()
        self.window.title("BERT Text Autocomplete")
        self.create_ui()  # Changed from setup_ui to create_ui
        
    def create_ui(self):  # New method
        # Input area
        self.input_area = scrolledtext.ScrolledText(self.window, width=60, height=10)
        self.input_area.pack(pady=10)
        self.input_area.bind('<KeyRelease>', self.on_key_release)
        
        # Suggestions area
        self.suggestions_label = tk.Label(self.window, text="Suggestions:")
        self.suggestions_label.pack()
        self.suggestions_area = scrolledtext.ScrolledText(self.window, width=60, height=5)
        self.suggestions_area.pack(pady=10)
    
    def on_key_release(self, event=None):
        # Get last complete word for better context
        text = self.input_area.get("1.0", tk.END).strip()
        if text:
            suggestions = self.model.get_suggestions(text)
            self.suggestions_area.delete("1.0", tk.END)
            if suggestions:
                self.suggestions_area.insert("1.0", "Suggestions: " + ", ".join(suggestions))
            else:
                self.suggestions_area.insert("1.0", "No valid suggestions")

    def run(self):
        try:
            self.window.mainloop()
        except Exception as e:
            print(f"Error in UI: {e}")

def load_training_data(tokenizer, max_samples=10000):
    print("Loading WikiText-103 dataset...")
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train[:10000]')
    
    print("Filtering and processing texts...")
    # Filter for high-quality writing samples
    filtered_texts = [
        text for text in dataset['text'] 
        if len(text.split()) > 10  # Longer sentences only
        and text.strip().endswith(('.', '!', '?'))  # Complete sentences
        and not text.startswith('=')  # Remove Wiki headers
        and len(text.strip()) > 0
    ]
    
    print(f"Number of training examples: {len(filtered_texts)}")
    return TextDataset({"text": filtered_texts}, tokenizer)

def train_model(model, train_dataset, device, epochs=3):
    print(f"Training on device: {device}")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    model.train()
    for epoch in range(epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        total_loss = 0
        
        for batch in progress_bar:
            try:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            except RuntimeError as e:
                print(f"Error during training: {e}")
                continue
            
        gc.collect()
        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} completed, Average Loss: {avg_loss:.4f}")

def main():
    print("Initializing with larger dataset...")
    model = EnhancedBERTAutocomplete()
    
    ui = TextAutocompleteUI(model)
    ui.run()

if __name__ == "__main__":
    main()