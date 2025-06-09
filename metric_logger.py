import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from training_utilities import macro_soft_roc_curve


class MetricLogger:
    def __init__(self, cfg):
        """Initialize the metric logger with configuration"""
        self.cfg = cfg
        self.metrics_df = pd.DataFrame()
        # Store best epoch ROC data for each fold
        self.best_roc_data = {}
        # Track best epoch per fold
        self.best_epochs = {}
        
    def log_metrics(self, epoch, fold, train_metrics, val_metrics, roc_data):
        """Log metrics for one epoch"""
        # Create a row with all metrics
        metrics_row = {
            'epoch': epoch + 1,  # Store as 1-indexed for consistency
            'fold': fold,
            'train_loss': train_metrics['train_loss'],
            'train_auc': train_metrics['train_auc'],
            'val_loss': val_metrics['val_loss'],
            'val_auc': val_metrics['val_auc'],
            'learning_rate': train_metrics['learning_rate']
        }
        
        # Append to dataframe
        self.metrics_df = pd.concat([self.metrics_df, pd.DataFrame([metrics_row])], ignore_index=True)
        
        # Update best epoch if applicable - simpler logic for clarity
        current_val_auc = val_metrics['val_auc']
        
        # Initial case - first epoch for this fold
        if fold not in self.best_epochs:
            self.best_epochs[fold] = epoch + 1  # Store as 1-indexed
            self.best_roc_data[fold] = roc_data
            
        # Better AUC than previous best
        elif current_val_auc > self.best_epochs.get(str(fold) + "_auc", 0):
            self.best_epochs[fold] = epoch + 1  # Store as 1-indexed
            self.best_epochs[str(fold) + "_auc"] = current_val_auc  # Also store the AUC value directly
            self.best_roc_data[fold] = roc_data
    
    def plot_metrics(self):
        """Generate plots for all metrics"""
        # Generate each plot
        self._plot_loss_curves()
        self._plot_auc_curves()
        self._plot_roc_curves()
        
        # Save metrics to CSV
        metrics_file = f"{self.cfg.metrics_dir}/metrics_{self.cfg.timestamp}.csv"
        self.metrics_df.to_csv(metrics_file, index=False)
        print(f"Metrics saved to {metrics_file}")
    
    def _plot_loss_curves(self):
        """Plot training and validation loss curves for all folds"""
        plt.figure(figsize=(12, 8))
        
        folds = sorted([f for f in self.best_epochs.keys() if not isinstance(f, str)])
        colors = plt.cm.tab10(np.linspace(0, 1, len(folds)))
        
        for i, fold in enumerate(folds):
            fold_data = self.metrics_df[self.metrics_df['fold'] == fold]
            color = colors[i]
            
            # Plot training loss
            plt.plot(fold_data['epoch'], fold_data['train_loss'], 
                     marker='o', linestyle='-', color=color, alpha=0.7,
                     label=f'Fold {fold} - Train Loss')
            
            # Plot validation loss
            plt.plot(fold_data['epoch'], fold_data['val_loss'], 
                     marker='s', linestyle='--', color=color,
                     label=f'Fold {fold} - Val Loss')
            
            # Mark best epoch
            if fold in self.best_epochs:
                best_epoch = self.best_epochs[fold]
                best_data = fold_data[fold_data['epoch'] == best_epoch]
                if not best_data.empty:
                    plt.scatter(best_epoch, best_data['val_loss'].values[0], 
                                marker='*', s=200, color=color, edgecolor='black',
                                label=f'Fold {fold} - Best Epoch' if i == 0 else "")
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        plt.savefig(f'{self.cfg.plots_dir}/loss_curves_{self.cfg.timestamp}.png')
    
    def _plot_auc_curves(self):
        """Plot training and validation AUC curves for all folds"""
        plt.figure(figsize=(12, 8))
        
        folds = sorted([f for f in self.best_epochs.keys() if not isinstance(f, str)])
        colors = plt.cm.tab10(np.linspace(0, 1, len(folds)))
        
        for i, fold in enumerate(folds):
            fold_data = self.metrics_df[self.metrics_df['fold'] == fold]
            color = colors[i]
            
            # Plot training AUC
            plt.plot(fold_data['epoch'], fold_data['train_auc'], 
                     marker='o', linestyle='-', color=color, alpha=0.7,
                     label=f'Fold {fold} - Train AUC')
            
            # Plot validation AUC
            plt.plot(fold_data['epoch'], fold_data['val_auc'], 
                     marker='s', linestyle='--', color=color,
                     label=f'Fold {fold} - Val AUC')
            
            # Mark best epoch
            if fold in self.best_epochs:
                best_epoch = self.best_epochs[fold]
                best_data = fold_data[fold_data['epoch'] == best_epoch]
                if not best_data.empty:
                    plt.scatter(best_epoch, best_data['val_auc'].values[0], 
                                marker='*', s=200, color=color, edgecolor='black',
                                label=f'Fold {fold} - Best Epoch' if i == 0 else "")
        
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.title('Training and Validation AUC')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        plt.savefig(f'{self.cfg.plots_dir}/auc_curves_{self.cfg.timestamp}.png')
    
    def _plot_roc_curves(self):
        """Plot ROC curves for best epoch of each fold"""
        plt.figure(figsize=(12, 8))
        
        # Check if we have any data to plot
        folds = [f for f in self.best_epochs.keys() if not isinstance(f, str)]
        if not folds:
            print("No best epochs found, skipping ROC plot")
            plt.close()
            return
        
        folds = sorted(folds)
        colors = plt.cm.tab10(np.linspace(0, 1, len(folds)))
        curves_plotted = 0
        
        print(f"Plotting ROC curves for folds: {folds}")
        
        for i, fold in enumerate(folds):
            if fold not in self.best_roc_data:
                print(f"No ROC data available for fold {fold}")
                continue
            
            # Safely extract data
            try:
                roc_data = self.best_roc_data[fold]
                if not isinstance(roc_data, tuple) or len(roc_data) != 2:
                    print(f"Fold {fold} has invalid ROC data format: {type(roc_data)}")
                    continue
                    
                targets, probs = roc_data
                
                # Handle NaN values in the probability array
                if np.isnan(probs).any():
                    print(f"Warning: NaN values found in fold {fold} predictions. Replacing with zeros.")
                    probs = np.nan_to_num(probs, nan=0.0)
                
                if np.isnan(targets).any():
                    print(f"Warning: NaN values found in fold {fold} targets. Replacing with zeros.")
                    targets = np.nan_to_num(targets, nan=0.0)
                
                # Get the best AUC value for this fold (directly from our stored value)
                best_auc = self.best_epochs.get(str(fold) + "_auc", 0)
                if best_auc == 0:
                    # Fall back to lookup if not stored directly
                    best_epoch = self.best_epochs[fold]
                    best_auc_rows = self.metrics_df[
                        (self.metrics_df["fold"] == fold) & 
                        (self.metrics_df["epoch"] == best_epoch)
                    ]
                    
                    if best_auc_rows.empty:
                        print(f"No metrics found for fold {fold}, epoch {best_epoch}")
                        continue
                        
                    best_auc = best_auc_rows["val_auc"].values[0]
                
                # Calculate average ROC curve across all classes
                fpr, tpr = macro_soft_roc_curve(targets, probs)
                
                # Check if we got a meaningful curve (not just the diagonal)
                if np.allclose(fpr, tpr, rtol=1e-4, atol=1e-4):
                    print(f"Warning: ROC curve for fold {fold} is a diagonal line (random performance)")
                
                # Plot ROC curve for this fold
                plt.plot(fpr, tpr, lw=2, color=colors[i],
                         label=f'Fold {fold} (AUC = {best_auc:.4f})')
                curves_plotted += 1
            except Exception as e:
                import traceback
                print(f"Error plotting ROC curve for fold {fold}: {e}")
                traceback.print_exc()
                continue
        
        if curves_plotted == 0:
            print("No valid ROC curves could be plotted. Skipping ROC curve generation.")
            plt.close()
            return
            
        # Add reference diagonal line
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Best Epochs')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()
        plt.savefig(f'{self.cfg.plots_dir}/roc_curves_{self.cfg.timestamp}.png')
