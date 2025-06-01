import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import torch
import torch.nn.functional as F
import gc 
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import interpolate
import time

def get_scheduler(optimizer, cfg, steps=None):
    if cfg.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.T_max,
            eta_min=cfg.min_lr
        )
    elif cfg.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=cfg.min_lr,
            verbose=True
        )
    elif cfg.scheduler == 'StepLR':
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.epochs // 3,
            gamma=0.5
        )
    elif cfg.scheduler == 'OneCycleLR':
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.lr,
            steps_per_epoch=steps,
            epochs=cfg.epochs,
            pct_start=0.1)
    else:
        scheduler = None   
    
    # Add warmup capability to existing schedulers
    if hasattr(cfg, 'use_lr_warmup') and cfg.use_lr_warmup and hasattr(cfg, 'warmup_epochs') and cfg.warmup_epochs > 0:
        if scheduler is not None:
            scheduler = GradualWarmupScheduler(
                optimizer, 
                multiplier=1, 
                total_epoch=cfg.warmup_epochs, 
                after_scheduler=scheduler
            )
            
    return scheduler

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingBCEWithLogitsLoss(nn.Module):
    """Custom implementation of BCE with label smoothing since BCEWithLogitsLoss doesn't support it natively"""
    def __init__(self, label_smoothing=0.0):
        super(LabelSmoothingBCEWithLogitsLoss, self).__init__()
        self.label_smoothing = label_smoothing
        
    def forward(self, pred, target):
        # Apply label smoothing: move target values away from 0 and 1
        if self.label_smoothing > 0:
            # Smooth the labels
            target = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        return F.binary_cross_entropy_with_logits(pred, target)

def get_criterion(cfg):
    if cfg.criterion == 'BCEWithLogitsLoss':
        if hasattr(cfg, 'label_smoothing') and cfg.label_smoothing > 0:
            criterion = LabelSmoothingBCEWithLogitsLoss(label_smoothing=cfg.label_smoothing)
        else:
            criterion = nn.BCEWithLogitsLoss()
    elif cfg.criterion == 'FocalLoss':
        criterion = FocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
    elif cfg.criterion == 'CombinedLoss':
        criterion = CombinedLoss(
            alpha=cfg.focal_alpha, 
            gamma=cfg.focal_gamma,
            bce_weight=cfg.bce_weight,
            focal_weight=cfg.focal_weight,
            label_smoothing=cfg.label_smoothing if hasattr(cfg, 'label_smoothing') else 0.0
        )
    else:
        raise NotImplementedError(f"Criterion {cfg.criterion} not implemented")
    return criterion

class CombinedLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, bce_weight=0.5, focal_weight=0.5, label_smoothing=0.0):
        super(CombinedLoss, self).__init__()
        self.label_smoothing = label_smoothing
        if label_smoothing > 0:
            self.bce_loss = LabelSmoothingBCEWithLogitsLoss(label_smoothing=label_smoothing)
        else:
            self.bce_loss = nn.BCEWithLogitsLoss()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        
    def forward(self, inputs, targets):
        bce = self.bce_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        return self.bce_weight * bce + self.focal_weight * focal

class GradualWarmupScheduler(_LRScheduler):
    """Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    """
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)
        
def clean_gpu_memory(cfg):
    """
    Thoroughly cleans GPU memory to prevent CUDA out-of-memory errors between folds
    """
    torch._dynamo.reset()
    
    # Delete all tensors from GPU memory
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.device.type == 'cuda':
                del obj
        except:
            pass
    
    # Force garbage collection multiple times
    for _ in range(3):
        gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # On some systems, this might help further
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_max_memory_allocated()
        except:
            pass

    time.sleep(1)  # Give time for CUDA to release memory


def calculate_soft_label_metrics(y_true, y_pred):
    """Calculate comprehensive metrics for soft label multi-label classification
    
    Args:
        y_true: True labels (can be soft labels between 0 and 1) 
        y_pred: Predicted probabilities
    """
    metrics = {}
    
    # Check for and handle NaN values
    if np.isnan(y_pred).any():
        print("Warning: NaN values detected in predictions. Replacing with zeros.")
        y_pred = np.nan_to_num(y_pred, nan=0.0)
    
    # Per-class AUCs for analysis
    class_aucs = []
    valid_classes = 0
    
    for class_idx in range(y_true.shape[1]):
        class_true = y_true[:, class_idx]
        class_pred = y_pred[:, class_idx]
        
        # Check for variation in true labels
        if np.var(class_true) > 0:
            try:
                # to calculate AUC, we first assume all labels are true
                # then we assume that all are false
                y_true_hard = np.concatenate((np.ones(class_true.shape), np.zeros(class_true.shape)))
                sample_weight = np.concatenate((class_true, 1-class_true))
                y_pred_twice = np.concatenate((class_pred, class_pred))
                
                auc_score = roc_auc_score(y_true_hard, y_pred_twice, sample_weight=sample_weight)
                class_aucs.append(auc_score)
                valid_classes += 1
            except Exception as e:
                print(f"Error calculating AUC for class {class_idx}: {e}")
                class_aucs.append(0.5)  # Default to random performance
        else:
            # If no variation, AUC is undefined - use 0.5 (random performance)
            class_aucs.append(0.5)
    
    metrics['macro_auc'] = np.mean(class_aucs)
    metrics['valid_classes'] = valid_classes
    
    return metrics

def macro_soft_roc_curve(y_true_soft, y_score, num_points=100):
    """
    Computes the macro-averaged ROC curve for multiclass soft labels.
    
    Parameters:
        y_true_soft: np.ndarray, shape (n_samples, n_classes)
            Soft ground truth labels (should sum to 1 per row).
        y_score: np.ndarray, shape (n_samples, n_classes)
            Predicted probabilities per class.
        num_points: int
            Number of FPR points to interpolate over (shared grid).
    
    Returns:
        fpr_grid: np.ndarray, shape (num_points,)
        mean_tpr: np.ndarray, shape (num_points,)
    """
    # Handle NaN values in predictions
    if np.isnan(y_score).any():
        print("Warning: NaN values detected in ROC curve calculation. Replacing with zeros.")
        y_score = np.nan_to_num(y_score, nan=0.0)
    
    # Handle NaN values in targets
    if np.isnan(y_true_soft).any():
        print("Warning: NaN values in targets detected. Replacing with zeros.")
        y_true_soft = np.nan_to_num(y_true_soft, nan=0.0)
    
    n_classes = y_true_soft.shape[1]
    fpr_grid = np.linspace(0, 1, num_points)
    all_tprs = []
    valid_classes = 0

    for i in range(n_classes):
        y_t = y_true_soft[:, i]
        y_s = y_score[:, i]
        
        try:
            y_true_bin = []
            y_score_bin = []
            sample_weight = []

            for yt, ys in zip(y_t, y_s):
                if yt > 0:
                    y_true_bin.append(1)
                    y_score_bin.append(ys)
                    sample_weight.append(yt)
                if yt < 1:
                    y_true_bin.append(0)
                    y_score_bin.append(ys)
                    sample_weight.append(1 - yt)

            y_true_bin = np.array(y_true_bin)
            y_score_bin = np.array(y_score_bin)
            sample_weight = np.array(sample_weight)
            
            # Skip if insufficient data
            if len(y_true_bin) == 0 or len(np.unique(y_true_bin)) < 2:
                continue

            fpr, tpr, _ = roc_curve(y_true_bin, y_score_bin, sample_weight=sample_weight)
            
            # Skip if not enough points for interpolation
            if len(fpr) < 2:
                continue

            # Make sure fpr is sorted for interpolation
            sort_idx = np.argsort(fpr)
            fpr = fpr[sort_idx]
            tpr = tpr[sort_idx]

            # Interpolate TPR to the common FPR grid
            # Add explicit points at 0 and 1 if they don't exist
            if fpr[0] > 0:
                fpr = np.concatenate([[0], fpr])
                tpr = np.concatenate([[0], tpr])
            if fpr[-1] < 1:
                fpr = np.concatenate([fpr, [1]])
                tpr = np.concatenate([tpr, [1]])
                
            interp_tpr = interpolate.interp1d(fpr, tpr, bounds_error=False, fill_value=(0, 1))
            interpolated_tpr = interp_tpr(fpr_grid)
            all_tprs.append(interpolated_tpr)
            valid_classes += 1
        except Exception as e:
            print(f"Error calculating ROC curve for class {i}: {e}")
            continue

    # If no valid curves, return a diagonal line (random classifier)
    if len(all_tprs) == 0:
        print(f"Warning: No valid ROC curves could be calculated out of {n_classes} classes. Returning diagonal line.")
        return fpr_grid, fpr_grid
        
    # Convert to array and calculate mean
    all_tprs = np.array(all_tprs)
    mean_tpr = np.mean(all_tprs, axis=0)
    
    # Ensure mean_tpr starts at 0 and ends at 1
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0
    
    return fpr_grid, mean_tpr


def calculate_hard_label_metrics(y_true, y_pred):
    """Calculate comprehensive metrics for binary hard label multi-label classification
    
    Args:
        y_true: True binary labels (0s and 1s)
        y_pred: Predicted probabilities
    """
    metrics = {}
    
    # Check for and handle NaN values
    if np.isnan(y_pred).any():
        print("Warning: NaN values detected in predictions. Replacing with zeros.")
        y_pred = np.nan_to_num(y_pred, nan=0.0)
    
    # Per-class AUCs for analysis
    class_aucs = []
    valid_classes = 0
    
    for class_idx in range(y_true.shape[1]):
        class_true = y_true[:, class_idx]
        class_pred = y_pred[:, class_idx]
        
        # Check if class exists in dataset
        if np.any(class_true > 0) and np.any(class_true < 1):
            try:
                auc_score = roc_auc_score(class_true, class_pred)
                class_aucs.append(auc_score)
                valid_classes += 1
            except Exception as e:
                print(f"Error calculating AUC for class {class_idx}: {e}")
    
    metrics['macro_auc'] = np.mean(class_aucs)
    metrics['valid_classes'] = valid_classes
    
    return metrics


def macro_hard_roc_curve(y_true, y_score, num_points=100):
    """
    Computes the macro-averaged ROC curve for multiclass hard (binary) labels.
    
    Parameters:
        y_true: np.ndarray, shape (n_samples, n_classes)
            Binary ground truth labels (0s and 1s).
        y_score: np.ndarray, shape (n_samples, n_classes)
            Predicted probabilities per class.
        num_points: int
            Number of FPR points to interpolate over (shared grid).
    
    Returns:
        fpr_grid: np.ndarray, shape (num_points,)
        mean_tpr: np.ndarray, shape (num_points,)
    """
    # Handle NaN values in predictions
    if np.isnan(y_score).any():
        print("Warning: NaN values detected in ROC curve calculation. Replacing with zeros.")
        y_score = np.nan_to_num(y_score, nan=0.0)
    
    n_classes = y_true.shape[1]
    fpr_grid = np.linspace(0, 1, num_points)
    all_tprs = []
    valid_classes = 0

    for i in range(n_classes):
        y_t = y_true[:, i]
        y_s = y_score[:, i]
        
        try:
            # Skip if insufficient data
            if len(np.unique(y_t)) < 2:
                continue

            fpr, tpr, _ = roc_curve(y_t, y_s)
            
            # Skip if not enough points for interpolation
            if len(fpr) < 2:
                continue

            # Make sure fpr is sorted for interpolation
            sort_idx = np.argsort(fpr)
            fpr = fpr[sort_idx]
            tpr = tpr[sort_idx]

            # Interpolate TPR to the common FPR grid
            # Add explicit points at 0 and 1 if they don't exist
            if fpr[0] > 0:
                fpr = np.concatenate([[0], fpr])
                tpr = np.concatenate([[0], tpr])
            if fpr[-1] < 1:
                fpr = np.concatenate([fpr, [1]])
                tpr = np.concatenate([tpr, [1]])
                
            interp_tpr = interpolate.interp1d(fpr, tpr, bounds_error=False, fill_value=(0, 1))
            interpolated_tpr = interp_tpr(fpr_grid)
            all_tprs.append(interpolated_tpr)
            valid_classes += 1
        except Exception as e:
            print(f"Error calculating ROC curve for class {i}: {e}")
            continue

    # If no valid curves, return a diagonal line (random classifier)
    if len(all_tprs) == 0:
        print(f"Warning: No valid ROC curves could be calculated out of {n_classes} classes. Returning diagonal line.")
        return fpr_grid, fpr_grid
        
    # Convert to array and calculate mean
    all_tprs = np.array(all_tprs)
    mean_tpr = np.mean(all_tprs, axis=0)
    
    # Ensure mean_tpr starts at 0 and ends at 1
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0
    
    return fpr_grid, mean_tpr


def compile_model(model, cfg):
    """Safely compile the model with fallback options"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping model compilation")
        return model
        
    try:
        compiled_model = torch.compile(
            model, 
            backend=cfg.compile_backend,
            mode=cfg.compile_mode,
            fullgraph=False  # Set to False for better compatibility
        )
        print(f"Model compiled successfully with backend '{cfg.compile_backend}', mode '{cfg.compile_mode}'")
        return compiled_model
    except Exception as e:
        print(f"Model compilation failed: {e}")
        try:
            # Try with a more conservative mode
            compiled_model = torch.compile(
                model,
                backend="eager",
                mode="reduce-overhead",
                fullgraph=False
            )
            print("Model compiled with fallback backend eager, mode: reduce-overhead")
            return compiled_model
        except Exception as e2:
            print(f"Fallback compilation also failed: {e2}")
            print("Using non-compiled model")
            return model