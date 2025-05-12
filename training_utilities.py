import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import gc 
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.metrics import roc_auc_score

def get_optimizer(model, cfg):
    if cfg.optimizer == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=0.9,
            weight_decay=cfg.weight_decay
        )
    else:
        raise NotImplementedError(f"Optimizer {cfg.optimizer} not implemented") 
    return optimizer

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
        
def clean_gpu_memory():
    """
    Thoroughly cleans GPU memory to prevent CUDA out-of-memory errors between folds
    """
    # Delete all tensors from GPU memory
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.device.type == 'cuda':
                del obj
        except:
            pass
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    # Force garbage collection multiple times
    for _ in range(3):
        gc.collect()
    
    # Print memory stats if available
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            free = total_memory - reserved
            
            print(f"GPU {i} Memory: Total {total_memory:.2f}GB | "
                  f"Reserved {reserved:.2f}GB | Allocated {allocated:.2f}GB | Free {free:.2f}GB")

def calculate_auc(targets, probs):
    aucs = []
    
    for i in range(targets.shape[1]):
        if np.sum(targets[:, i]) > 0:
            class_auc = roc_auc_score(targets[:, i], probs[:, i])
            aucs.append(class_auc)
    
    return np.mean(aucs) if aucs else 0.0

def compile_model(model, cfg):
    """Safely compile the model with fallback options"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping model compilation")
        return model
        
    try:
        compiled_model = torch.compile(
            model, 
            backend="inductor",
            mode=cfg.compile_mode,
            fullgraph=False  # Set to False for better compatibility
        )
        print(f"Model compiled successfully with mode: {cfg.compile_mode}")
        return compiled_model
    except Exception as e:
        print(f"Model compilation failed: {e}")
        if cfg.compile_fallback:
            try:
                # Try with a more conservative mode
                compiled_model = torch.compile(
                    model,
                    backend="inductor",
                    mode="reduce-overhead",
                    fullgraph=False
                )
                print("Model compiled with fallback mode: reduce-overhead")
                return compiled_model
            except Exception as e2:
                print(f"Fallback compilation also failed: {e2}")
                print("Using non-compiled model")
                return model
        return model