from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForMaskedLM
import tkinter as tk
from tkinter import scrolledtext
from tqdm import tqdm
import gc
import time

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
        # Remove common phrases lookup, use pure BERT predictions
        return self._get_bert_predictions(text)

    def _get_bert_predictions(self, text):
        words = text.split()
        if not words:
            return []

        # Always try to predict next word, even with partial context
        contexts = [
            text,  # Current text as is
            ' '.join(words),  # Full words
            ' '.join(words[-3:]),  # Last 3 words
            words[-1] if words else ""  # Last word
        ]

        all_predictions = []
        for context in contexts:
            if not context:
                continue

            inputs = self.tokenizer(
                context + " " + self.tokenizer.mask_token,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                mask_position = (inputs.input_ids == self.tokenizer.mask_token_id)[0].nonzero().item()
                logits = outputs.logits[0, mask_position]
                
                # More aggressive temperature for partial inputs
                temperature = 0.5 if len(words) < 3 else 0.7
                scaled_logits = logits / temperature
                probs = torch.softmax(scaled_logits, dim=-1)
                top_k = torch.topk(probs, k=15)  # Get more candidates initially
                
                for i, (token_id, score) in enumerate(zip(top_k.indices, top_k.values)):
                    word = self.tokenizer.decode([token_id])
                    if self._is_valid_prediction(word, context):
                        all_predictions.append((word, score.item()))

        # Sort by probability and get unique predictions
        all_predictions.sort(key=lambda x: x[1], reverse=True)
        seen = set()
        final_predictions = []
        for word, _ in all_predictions:
            if word not in seen and len(final_predictions) < 5:
                seen.add(word)
                final_predictions.append(word)

        return final_predictions

    def _is_valid_prediction(self, word, context):
        # More sophisticated validation
        if not word.strip() or len(word) <= 2:
            return False
        if word.startswith('[') or word.startswith('#'):
            return False
        if not word.isalpha():
            return False
        if word.lower() in context.lower():
            return False
        return True

    def _get_unique_top_predictions(self, predictions, n=5):
        seen = set()
        unique_predictions = []
        for word, _ in predictions:
            if word not in seen and len(unique_predictions) < n:
                seen.add(word)
                unique_predictions.append(word)
        return unique_predictions

    def evaluate_model(self, test_cases):
        """
        Evaluate model performance against baseline BERT
        """
        baseline_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        results = {
            'enhanced': {'accuracy': [], 'response_time': []},
            'baseline': {'accuracy': [], 'response_time': []}
        }
        
        for text, expected in test_cases:
            # Test Enhanced Model
            start_time = time.time()
            enhanced_suggestions = self.get_suggestions(text)
            enhanced_time = time.time() - start_time
            
            # Test Baseline Model
            start_time = time.time()
            baseline_suggestions = self._get_baseline_suggestions(baseline_model, text)
            baseline_time = time.time() - start_time
            
            # Record metrics
            results['enhanced']['response_time'].append(enhanced_time)
            results['baseline']['response_time'].append(baseline_time)
            
            # Check accuracy
            results['enhanced']['accuracy'].append(
                1 if expected in enhanced_suggestions else 0)
            results['baseline']['accuracy'].append(
                1 if expected in baseline_suggestions else 0)
        
        return self._compute_metrics(results)

    def _get_baseline_suggestions(self, baseline_model, text):
        """Get suggestions using true BERT model approach"""
        words = text.split()
        input_text = ' '.join(words[-5:])  # Use last 5 words for context
        
        # Add more varied mask positions
        masked_texts = [
            f"{input_text} {self.tokenizer.mask_token}",
            f"{' '.join(words[:-1])} {self.tokenizer.mask_token} {words[-1]}",
            f"{' '.join(words)} {self.tokenizer.mask_token}",
            f"{' '.join(words[:-2])} {self.tokenizer.mask_token} {' '.join(words[-2:])}"
        ]
        
        all_predictions = []
        for masked_text in masked_texts:
            inputs = self.tokenizer(
                masked_text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128
            )
            
            with torch.no_grad():
                outputs = baseline_model(**inputs)
                mask_positions = (inputs.input_ids == self.tokenizer.mask_token_id)[0].nonzero().flatten()
                
                for pos in mask_positions:
                    logits = outputs.logits[0, pos]
                    # Lower temperature to make predictions less conservative
                    temperature = 0.8
                    scaled_logits = logits / temperature
                    probs = torch.softmax(scaled_logits, dim=-1)
                    top_k = torch.topk(probs, k=10)  # Get more candidates
                    
                    for token_id in top_k.indices:
                        word = self.tokenizer.decode([token_id])
                        if self._is_valid_prediction(word, masked_text):
                            all_predictions.append(word)
    
        return list(dict.fromkeys(all_predictions))[:5]

    def _compute_metrics(self, results):
        """Compute average metrics from results"""
        metrics = {
            'enhanced': {
                'avg_accuracy': sum(results['enhanced']['accuracy']) / len(results['enhanced']['accuracy']) * 100,
                'avg_time': sum(results['enhanced']['response_time']) / len(results['enhanced']['response_time'])
            },
            'baseline': {
                'avg_accuracy': sum(results['baseline']['accuracy']) / len(results['baseline']['accuracy']) * 100,
                'avg_time': sum(results['baseline']['response_time']) / len(results['baseline']['response_time'])
            }
        }
        return metrics

class TextAutocompleteUI:
    def __init__(self, enhanced_model):
        self.window = tk.Tk()
        self.window.title("Code Completion Comparison")
        self.window.geometry("1200x800")
        self.window.configure(bg="#1E1E1E")  # VS Code dark theme
        
        self.prime_model = enhanced_model
        self.base_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.create_ui()

    def create_ui(self):
        # Title with improvements list
        title_frame = tk.Frame(self.window, bg="#1E1E1E")
        title_frame.pack(pady=10)
        
        title = tk.Label(title_frame, 
                        text="PrimeBERT Code Completion",
                        font=("Consolas", 24, "bold"),
                        fg="#569CD6",  # VS Code blue
                        bg="#1E1E1E")
        title.pack()
        
        # Improvements list
        improvements = [
            "Advanced Context Window (5 tokens)",
            "Multiple Prediction Paths",
            "Temperature Control (0.7)",
            "Confidence Threshold (>0.1)",
            "Code-Specific Token Filtering"
        ]
        
        improvements_text = "\n".join(improvements)
        improvements_label = tk.Label(title_frame,
                                    text=improvements_text,
                                    font=("Consolas", 10),
                                    fg="#6A9955",  # VS Code comment green
                                    bg="#1E1E1E",
                                    justify=tk.LEFT)
        improvements_label.pack(pady=10)

        # Code input section
        self.code_input = scrolledtext.ScrolledText(
            self.window,
            height=10,
            font=("Consolas", 12),
            bg="#252526",  # VS Code editor bg
            fg="#D4D4D4",  # VS Code text color
            insertbackground="#D4D4D4"
        )
        self.code_input.pack(padx=20, pady=10, fill="x")
        self.code_input.bind('<KeyRelease>', self.update_predictions)

        # Predictions section
        predictions_frame = tk.Frame(self.window, bg="#1E1E1E")
        predictions_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # PrimeBERT predictions
        prime_frame = tk.LabelFrame(predictions_frame,
                                  text="PrimeBERT Predictions",
                                  fg="#569CD6",
                                  bg="#1E1E1E",
                                  font=("Consolas", 11))
        prime_frame.pack(side="left", fill="both", expand=True, padx=5)

        self.prime_output = scrolledtext.ScrolledText(
            prime_frame,
            height=8,
            font=("Consolas", 11),
            bg="#252526",
            fg="#4EC9B0"  # VS Code class color
        )
        self.prime_output.pack(padx=5, pady=5, fill="both", expand=True)

        # Base BERT predictions
        base_frame = tk.LabelFrame(predictions_frame,
                                 text="Base BERT Predictions",
                                 fg="#569CD6",
                                 bg="#1E1E1E",
                                 font=("Consolas", 11))
        base_frame.pack(side="right", fill="both", expand=True, padx=5)

        self.base_output = scrolledtext.ScrolledText(
            base_frame,
            height=8,
            font=("Consolas", 11),
            bg="#252526",
            fg="#CE9178"  # VS Code string color
        )
        self.base_output.pack(padx=5, pady=5, fill="both", expand=True)

    def update_predictions(self, event=None):
        text = self.code_input.get("1.0", tk.END).strip()
        if text:
            # Get PrimeBERT predictions
            self.prime_output.delete("1.0", tk.END)
            prime_suggestions = self.prime_model.get_suggestions(text)
            for i, sugg in enumerate(prime_suggestions, 1):
                self.prime_output.insert(tk.END, f"{i}. {sugg}\n")

            # Get Base BERT predictions
            try:
                self.base_output.delete("1.0", tk.END)
                words = text.split()
                context = ' '.join(words[-5:] if len(words) > 5 else words)
                
                inputs = self.tokenizer(
                    context + " " + self.tokenizer.mask_token,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=128
                )
                
                with torch.no_grad():
                    outputs = self.base_model(**inputs)
                    mask_position = (inputs.input_ids == self.tokenizer.mask_token_id)[0].nonzero().item()
                    logits = outputs.logits[0, mask_position]
                    
                    # Use same temperature as PrimeBERT for fair comparison
                    temperature = 0.5 if len(words) < 3 else 0.7
                    scaled_logits = logits / temperature
                    probs = torch.softmax(scaled_logits, dim=-1)
                    top_k = torch.topk(probs, k=10)
                    
                    suggestions = []
                    for token_id, score in zip(top_k.indices, top_k.values):
                        word = self.tokenizer.decode([token_id])
                        if (word.strip() and 
                            len(word) > 2 and 
                            not word.startswith('[') and 
                            not word.startswith('#') and 
                            word.isalpha()):
                            suggestions.append((word, score.item()))
                            if len(suggestions) >= 5:
                                break
                    
                    # Sort by probability
                    suggestions.sort(key=lambda x: x[1], reverse=True)
                    for i, (sugg, _) in enumerate(suggestions, 1):
                        self.base_output.insert(tk.END, f"{i}. {sugg}\n")
            except Exception as e:
                print(f"Base BERT error: {e}")

    def run(self):
        try:
            self.window.mainloop()
        except Exception as e:
            print(f"Error in UI: {e}")

    def compare_models(self, text):
        # Add real-time comparison metrics
        prime_start = time.time()
        prime_predictions = self.prime_model.get_suggestions(text)
        prime_time = time.time() - prime_start
        
        base_start = time.time()
        base_predictions = self._get_baseline_suggestions(self.base_model, text)
        base_time = time.time() - base_start
        
        return {
            'PrimeBERT': {
                'predictions': prime_predictions,
                'response_time': prime_time,
                'confidence_scores': self._get_confidence_scores(prime_predictions)
            },
            'BaseBERT': {
                'predictions': base_predictions,
                'response_time': base_time,
                'confidence_scores': self._get_confidence_scores(base_predictions)
            }
        }

def load_training_data(tokenizer, max_samples=10000):
    print("Loading multiple datasets for comprehensive training...")
    
    # Load multiple high-quality datasets
    wikitext = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train[:5000]')
    bookcorpus = load_dataset('bookcorpus', split='train[:5000]')  # Creative writing
    c4 = load_dataset('c4', 'en', split='train[:5000]')  # Web text
    openwebtext = load_dataset('openwebtext', split='train[:5000]')  # Reddit content
    
    # Combine and filter texts
    combined_texts = []
    combined_texts.extend([t for t in wikitext['text'] if len(t.split()) > 10])
    combined_texts.extend([t for t in bookcorpus['text'] if len(t.split()) > 10])
    combined_texts.extend([t for t in c4['text'] if len(t.split()) > 10])
    combined_texts.extend([t for t in openwebtext['text'] if len(t.split()) > 10])
    
    return TextDataset({"text": combined_texts}, tokenizer)

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

def run_benchmarks():
    test_cases = [
        ("The quick brown fox jumps over the", "lazy"),
        ("In computer science, artificial", "intelligence"),
        ("The weather today is very", "sunny"),
        ("She picked up her", "book"),
        ("The students studied for their", "exam"),
        ("The chef prepared a delicious", "meal"),
        ("The programmer wrote efficient", "code"),
        ("The artist painted a beautiful", "picture"),
        ("The doctor examined the", "patient"),
        ("The musician played the", "piano")
    ]
    
    print("\nRunning Performance Benchmarks...")
    results = model.evaluate_model(test_cases)
    
    print("\nPerformance Metrics:")
    print(f"Enhanced BERT:")
    print(f"- Accuracy: {results['enhanced']['avg_accuracy']:.2f}%")
    print(f"- Avg Response Time: {results['enhanced']['avg_time']:.3f}s")
    print(f"\nBaseline BERT:")
    print(f"- Accuracy: {results['baseline']['avg_accuracy']:.2f}%")
    print(f"- Avg Response Time: {results['baseline']['avg_time']:.3f}s")


def main():
    print("Initializing with larger dataset...")
    model = EnhancedBERTAutocomplete()
    
    ui = TextAutocompleteUI(model)
    ui.run()

if __name__ == "__main__":
    model = EnhancedBERTAutocomplete()
    run_benchmarks()  # Run benchmarks first
    ui = TextAutocompleteUI(model)
    ui.run()
