import math
import warnings
from typing import Dict, Literal
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")
import delu  # Deep Learning Utilities: https://github.com/Yura52/delu
import numpy as np
import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn.functional as F
import torch.optim
from torch import Tensor
from tqdm.std import tqdm
import pandas as pd

warnings.resetwarnings()
from rtdl_revisiting_models import MLP, ResNet, FTTransformer

# Add these imports for additional metrics
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set random seeds in all libraries.
delu.random.seed(0)

#Load the file
df_combined=pd.read_excel(input_file)

#selected features
df_combined=df_combined[['cause of infertility - partner sperm motility', 'embryos transfered', 'type of infertility - female secondary',
'cause of infertility - female factors', 'cause of infertility - partner sperm concentration', 'type of infertility - male secondary',
'type of infertility - male primary', 'live birth occurrence']]


df_combined = df_combined[df_combined['live birth occurrence'].isin([1, 0])]
df_combined['live birth occurrence']=df_combined['live birth occurrence'].astype(int)


# Define features before encoding because later after encoding no object-type columns may remain
catfeatures = df_combined.select_dtypes(include=['object']).columns.tolist()
numfeatures = df_combined.drop(['live birth occurrence'] + catfeatures, axis=1).columns.tolist()





#Encode Categorical Features
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

# Function to encode categorical columns
def encode_columns(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            # Convert all values to strings to ensure uniformity
            df[col] = df[col].astype(str)
            # Fit and transform the column
            df[col] = label_encoder.fit_transform(df[col])
    return df

# Apply the function to each dataframe copy
df_combined = encode_columns(df_combined)



# Define task type
TaskType = Literal["regression", "binclass", "multiclass"]
task_type: TaskType = "binclass"
n_classes = 2
assert n_classes >= 2
target = 'live birth occurrence' 

# Extract target variable
Y: np.ndarray = df_combined[target].values
# Continuous features
X_cont: np.ndarray = df_combined[numfeatures].to_numpy()
X_cont: np.ndarray = X_cont.astype(np.float32)
n_cont_features = X_cont.shape[1]

# Categorical features
X_cat: np.ndarray = df_combined[catfeatures].to_numpy().astype(np.int64)

# Get cardinalities after encoding
cat_cardinalities = []
if catfeatures:
    for feature in catfeatures:
        cat_cardinalities.append(df_combined[feature].nunique())

# >>> Labels.
if task_type == "regression":
    Y = Y.astype(np.float32)
else:
    assert n_classes is not None
    Y = Y.astype(np.int64)
    assert set(Y.tolist()) == set(
        range(n_classes)
    ), "Classification labels must form the range [0, 1, ..., n_classes - 1]"

# >>> Split the dataset with stratification for better balance
from sklearn.model_selection import train_test_split

all_idx = np.arange(len(Y))
trainval_idx, test_idx = train_test_split(
    all_idx, test_size=0.2, stratify=Y, random_state=42
)
train_idx, val_idx = train_test_split(
    trainval_idx, test_size=0.25, stratify=Y[trainval_idx], random_state=42  # 0.25 * 0.8 = 0.2 of total
)

data_numpy = {
    "train": {"x_cont": X_cont[train_idx], "y": Y[train_idx]},
    "val": {"x_cont": X_cont[val_idx], "y": Y[val_idx]},
    "test": {"x_cont": X_cont[test_idx], "y": Y[test_idx]},
}
if X_cat is not None:
    data_numpy["train"]["x_cat"] = X_cat[train_idx]
    data_numpy["val"]["x_cat"] = X_cat[val_idx]
    data_numpy["test"]["x_cat"] = X_cat[test_idx]

print(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}, Test size: {len(test_idx)}")
print(f"Train class distribution: {np.bincount(Y[train_idx])}")
print(f"Val class distribution: {np.bincount(Y[val_idx])}")
print(f"Test class distribution: {np.bincount(Y[test_idx])}")

# >>> Improved Feature preprocessing with outlier handling
from sklearn.preprocessing import RobustScaler, PowerTransformer

X_cont_train_numpy = data_numpy["train"]["x_cont"]

# Option A: Robust scaling (less sensitive to outliers)
# preprocessing = RobustScaler().fit(X_cont_train_numpy)

# Option B: Power transformer for better normality
# preprocessing = PowerTransformer(method='yeo-johnson', standardize=True).fit(X_cont_train_numpy)

# Option C: Quantile transformer with less aggressive normalization
noise = (
    np.random.default_rng(0)
    .normal(0.0, 1e-6, X_cont_train_numpy.shape)  # Reduced noise
    .astype(X_cont_train_numpy.dtype)
)
preprocessing = sklearn.preprocessing.QuantileTransformer(
    n_quantiles=min(len(train_idx) // 10, 500),  # Reduced quantiles for smoother transformation
    output_distribution="uniform",  # Try uniform instead of normal
    subsample=10**9,
).fit(X_cont_train_numpy + noise)
del X_cont_train_numpy

for part in data_numpy:
    data_numpy[part]["x_cont"] = preprocessing.transform(data_numpy[part]["x_cont"])

# >>> Label preprocessing for regression
if task_type == "regression":
    Y_mean = data_numpy["train"]["y"].mean().item()
    Y_std = data_numpy["train"]["y"].std().item()
    for part in data_numpy:
        data_numpy[part]["y"] = (data_numpy[part]["y"] - Y_mean) / Y_std

# >>> Convert data to tensors.
data = {
    part: {k: torch.as_tensor(v, device=device) for k, v in data_numpy[part].items()}
    for part in data_numpy
}

if task_type != "multiclass":
    for part in data:
        data[part]["y"] = data[part]["y"].float()

# Calculate gentler class weights
def calculate_gentle_class_weights(y_train):
    """Calculate gentler class weights using square root of inverse frequency"""
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    
    # Gentler weighting - square root of inverse frequency
    weights = np.sqrt(total_samples / (len(class_counts) * class_counts))
    # Further dampen the weights
    weights = (weights + 1.0) / 2.0  # Average with 1.0 to make weights less extreme
    return weights

class_weights_array = calculate_gentle_class_weights(data_numpy["train"]["y"])
class_weights_dict = {i: weight for i, weight in enumerate(class_weights_array)}
class_weights = torch.tensor(class_weights_array, dtype=torch.float32, device=device)

print(f"Gentle class weights dictionary: {class_weights_dict}")
print(f"Gentle class weights array: {class_weights_array}") 

# The output size.
d_out = n_classes if task_type == "multiclass" else 1

# Model selection with improved configurations
model_type = "fttransformer"  
# Print available default kwargs to see what can be customized
    default_kwargs = FTTransformer.get_default_kwargs()
    print(f"Available FTTransformer parameters: {list(default_kwargs.keys())}")
    print(f"Default values: {default_kwargs}")
    
    # Use default FTTransformer configuration - can be customized once we see available params
    model = FTTransformer(
        n_cont_features=n_cont_features,
        cat_cardinalities=cat_cardinalities,
        d_out=d_out,
        **FTTransformer.get_default_kwargs(),
    ).to(device)

print(f"Using model: {model_type}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Improved optimizer with lower learning rate
if model_type == "fttransformer":
    # Use FTTransformer's default optimizer but with modified parameters
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=1e-5,  # Much lower learning rate
        weight_decay=1e-4,  # Increased weight decay
        eps=1e-8,
        betas=(0.9, 0.999)
    )
else:
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=1e-5,  # Much lower learning rate
        weight_decay=1e-4,  # Increased weight decay
        eps=1e-8
    )

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,  # Initial period
    T_mult=2,  # Period multiplier
    eta_min=1e-8  # Minimum learning rate
)
def apply_model(batch: Dict[str, Tensor]) -> Tensor:
    if isinstance(model, (MLP, ResNet)):
        x_cat_ohe = (
            [
                F.one_hot(column, cardinality)
                for column, cardinality in zip(batch["x_cat"].T, cat_cardinalities)
            ]
            if "x_cat" in batch
            else []
        )
        return model(torch.column_stack([batch["x_cont"]] + x_cat_ohe)).squeeze(-1)

    elif isinstance(model, FTTransformer):
        return model(batch["x_cont"], batch.get("x_cat")).squeeze(-1)

    else:
        raise RuntimeError(f"Unknown model type: {type(model)}")

# Focal Loss implementation for better handling of hard examples
class FocalLoss:
    def __init__(self, alpha=None, gamma=2.0):
        self.alpha = alpha  # class weights
        self.gamma = gamma  # focusing parameter
    
    def __call__(self, predictions, targets):
        # Ensure predictions and targets have the same shape
        targets = targets.squeeze()  # Remove extra dimensions
        predictions = predictions.squeeze()  # Remove extra dimensions
        
        # Convert predictions to probabilities
        probs = torch.sigmoid(predictions)
        
        # Binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        
        # Focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha[1] * targets + self.alpha[0] * (1 - targets)
            focal_loss = alpha_t * focal_weight * bce_loss
        else:
            focal_loss = focal_weight * bce_loss
        
        return focal_loss.mean()

# Initialize loss function
focal_loss = FocalLoss(alpha=class_weights, gamma=1.5)  # Gentler gamma

def improved_loss_fn(predictions, targets):
    if task_type == "binclass":
        # Use focal loss for better handling of hard examples
        return focal_loss(predictions, targets)
    elif task_type == "multiclass":
        return F.cross_entropy(predictions, targets, weight=class_weights)
    else:  # regression
        return F.mse_loss(predictions, targets)

# Enhanced evaluation with additional metrics
@torch.no_grad()
def evaluate(part: str, return_detailed=False):
    model.eval()

    eval_batch_size = 1024  # Smaller batch size for stability
    y_pred = (
        torch.cat(
            [
                apply_model(batch)
                for batch in delu.iter_batches(data[part], eval_batch_size)
            ]
        )
        .cpu()
        .numpy()
    )
    y_true = data[part]["y"].cpu().numpy()

    if task_type == "binclass":
        y_pred_proba = scipy.special.expit(y_pred)
        y_pred_binary = (y_pred_proba >= 0.5).astype(int)
        
        accuracy = sklearn.metrics.accuracy_score(y_true, y_pred_binary)
        
        if return_detailed:
            auc = sklearn.metrics.roc_auc_score(y_true, y_pred_proba)
            precision = sklearn.metrics.precision_score(y_true, y_pred_binary)
            recall = sklearn.metrics.recall_score(y_true, y_pred_binary)
            f1 = sklearn.metrics.f1_score(y_true, y_pred_binary)
            
            # Add new metrics
            auprc = sklearn.metrics.average_precision_score(y_true, y_pred_proba)
            kappa = cohen_kappa_score(y_true, y_pred_binary)
            mcc = matthews_corrcoef(y_true, y_pred_binary)
            
            return {
                'accuracy': accuracy,
                'auc': auc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auprc': auprc,
                'kappa': kappa,
                'mcc': mcc
            }
        
        return accuracy
    elif task_type == "multiclass":
        y_pred = y_pred.argmax(1)
        score = sklearn.metrics.accuracy_score(y_true, y_pred)
    else:
        assert task_type == "regression"
        score = -(sklearn.metrics.mean_squared_error(y_true, y_pred) ** 0.5)
    return score

print(f'Test score before training: {evaluate("test"):.4f}')



# Training configuration
n_epochs = 1000  # Reduced for faster experimentation
patience = 30    # Increased patience
batch_size = 512  # Smaller batch size for more stable gradients
gradient_accumulation_steps = 4  # Effective batch size = 512 * 4 = 2048
actual_batch_size = batch_size // gradient_accumulation_steps  # 128
epoch_size = math.ceil(len(train_idx) / batch_size)
timer = delu.tools.Timer()
early_stopping = delu.tools.EarlyStopping(patience, mode="max")
best = {
    "val": -math.inf,
    "test": -math.inf,
    "epoch": -1,
}








