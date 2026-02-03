



# Transformer Model for Neural Machine Translation (NMT)

This project implements the foundational **Transformer architecture** (Vaswani et al., 2017) using PyTorch for a Sequence-to-Sequence (Seq2Seq) task, specifically **Neural Machine Translation (NMT)** on the Multi30k dataset (English -> German).

---

## Project Structure

The code is organized into several modules for clarity and reusability:

| File/Directory      | Description |
| :---                | :--- |
| TransformerTop.py   | **Main execution script** (entry point). Handles training, evaluation, and **inline epoch time calculation**. |
| TransformerBasic.py | Defines the main **Transformer(nn.Module)** class, connecting the Encoder and Decoder. |
| EncDecLayer.py      | Defines the **Encoder**, **Decoder**, **EncoderLayer**, and **DecoderLayer** modules. |
| Attention.py        | Contains the **MultiHeadAttention** and **FFN** (Feed-Forward Network) implementations. |
| LayerNorm.py        | Defines the custom **LayerNorm** utility. |
| Embeddings.py       | Defines the **TransformerEmbedding** class (token embedding + positional encoding). |
| DatasetLoads.py     | Handles loading and preprocessing of the Multi30k dataset using legacy **torchtext.data**. |
| Config.py           | Contains all global hyperparameters (e.g., **d_model**, **n_layers**, **init_lr**, **clip**, etc.). |
| bleu.py             | Contains functions (get_bleu, idx_to_word) for calculating the **BLEU metric**. |
| CheckpointManager.py| Checkpoint management utilities for saving and loading model states. |
| ModelSetupManager.py| Centralized model, vocabulary, and data loader initialization. |
| ConfigSanityCheck.py| Validates configuration parameters and resets invalid values to defaults. |
| saved/              | Directory for storing and loading model checkpoints (created automatically). |


---

## Setup and Installation

### 1. Environment and Dependencies

This project requires Python 3.9+ and is configured to utilize the **MPS (Metal Performance Shaders)** backend on Apple Silicon if available.

```bash
# Recommended: Create and activate a dedicated environment
conda create -n nmt_transformer python=3.9
conda activate nmt_transformer

# Install core dependencies
pip install torch torchvision torchaudio torchtext numpy requests

# Install spaCy for tokenization
pip install spacy

# Download language models required by DatasetLoads.py
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm


#Download sacrebleu for BLEU scores
pip install sacrebleu

#Download regex 
pip install regex



### 2. Data Setup

The data loader expects the Multi30k files to be accessible. If the automatic download fails, you must ensure the following raw text files are placed directly inside the data directory required by your script:

train.en, train.de

val.en, val.de

test2016.en, test2016.de


---


## Configuration (Config.py) Details
All critical settings for the model, optimizer, and training loop are centralized in Config.py.

Model Architecture Parameters:

| Parameter        | Value | Description |
| :---             | :---  | :---  |
| batch_size       | 64    | Number of samples processed per iteration | 
| max_len          | 128   | Maximum sequence length for padding/truncation |
| d_model          | 512   | Embedding/Hidden dimension |
| n_layers         | 6     | Number of Encoder/Decoder layers |
| n_heads          | 8     | Number of attention heads (divides d_model) |
| ffn_hidden       | 2048  | Intermediate dimension of the FFN layer (4 x d_model) |
| drop_prob        | 0.1   | Dropout probability (consider increasing to 0.3 if overfitting occurs) |


Optimizer and training control:

| Parameter        | Value  | Description |
| :---             | :---   | :---  |
| init_lr          | 1e-4   | Initial learning rate for the Adam optimizer |
| clip             | 1.0    | Gradient norm clipping value |
| factor           | 0.1    | LR reduction factor for ReduceLROnPlateau |
| patience         | 3      | Epochs to wait before reducing LR if validation loss stalls |
| warmup           | 4000   | Steps for the initial learning rate warmup |
| epoch            | 50     | Maximum number of training epochs |
| weight_decay     | 1e-4   | L2 regularization applied during optimization |
| label_smoothing  | 0.1    | Regularization factor to prevent overconfidence |
| resume_training  | True   | Continues a training session from the last best checkpoint (val loss) |
| strict_config    | True   | Enforces exact matches between critical parameters in your current settings and the checkpoint settings to prevent errors |


---

## Evaluation and Metrics:

The script outputs three primary metrics after each epoch's validation phase:

| Metric     | Goal     | Interpretation | 
| :---       | :---     | :---  |
| Val Loss   | Minimize | The most reliable measure of learning. Should decrease steadily. If it starts to increase, the model is overfitting and training should stop. |
| Val PPL    | Minimize | Perplexity (e^Loss). A lower value indicates the model is less ""surprised"" by the correct next word. |
| BLEU Score | Maximize | The primary quality metric for translation. Measures n-gram overlap with reference translations. A score of 25.0 or higher is often considered good for this dataset. |

The train_loss, val_loss and bleu scores for the latest run can be found in saved/ folder. 

---

## Running the training
Ensure all necessary files (Config.py, TransformerBasic.py, etc.) are in the same directory.

Run using: python TransformerTop.py

The script will automatically check for the existence of the saved/ directory and create it if necessary.



