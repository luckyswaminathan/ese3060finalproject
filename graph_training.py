import re
import glob
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_log_file(filepath):
    """Parse a log file and extract training metrics for each run."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    runs = {}
    current_run = None
    
    for line in lines:
        # Skip header lines and separators
        if 'run' in line.lower() and 'epoch' in line.lower():
            continue
        if line.strip().startswith('---'):
            continue
        
        # Match lines with data: | run | epoch | train_loss | train_acc | val_acc | ...
        # Run number can be blank (spaces) or a number
        # Pattern: | (run_num or spaces) | epoch | train_loss | train_acc | val_acc | ...
        match = re.match(r'\|\s+(\d+|\s+)\s+\|\s+(\d+|eval)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)', line)
        
        if match:
            run_str = match.group(1).strip()
            epoch = match.group(2)
            train_loss = float(match.group(3))
            train_acc = float(match.group(4))
            val_acc = float(match.group(5))
            
            # Update current run if a new run number is present
            if run_str:
                current_run = int(run_str)
            
            # Skip if we don't have a current run yet
            if current_run is None:
                continue
            
            # Skip eval rows
            if epoch == 'eval':
                continue
            
            epoch = int(epoch)
            
            if current_run not in runs:
                runs[current_run] = {'epochs': [], 'train_loss': [], 'train_acc': [], 'val_acc': []}
            
            runs[current_run]['epochs'].append(epoch)
            runs[current_run]['train_loss'].append(train_loss)
            runs[current_run]['train_acc'].append(train_acc)
            runs[current_run]['val_acc'].append(val_acc)
    
    return runs

def plot_training_flow(log_dir='logs_speed_baseline', num_runs=5, output_name=None):
    """Plot training flow for the first N runs, one plot per run."""
    # Find all log files
    log_files = sorted(glob.glob(f'{log_dir}/airbench94_baseline_run*.txt'))
    
    if not log_files:
        # Try time files if run files don't exist
        log_files = sorted(glob.glob(f'{log_dir}/airbench94_baseline_time*.txt'))
    
    if not log_files:
        print(f"No log files found in {log_dir}")
        return
    
    print(f"Found {len(log_files)} log files in {log_dir}")
    
    # Parse the first log file (or combine all if needed)
    # For now, let's use the first file which should have all runs
    all_runs = parse_log_file(log_files[0])
    
    if not all_runs:
        print("No data found in log files")
        return
    
    print(f"Found {len(all_runs)} runs in log file")
    
    # Determine title and output filename based on directory
    if 'baseline' in log_dir.lower():
        title = 'Training Flow - Baseline (Individual Runs)'
        if output_name is None:
            output_name = 'training_flow_baseline.png'
    elif 'changes' in log_dir.lower():
        title = 'Training Flow - Modified Version (Individual Runs)'
        if output_name is None:
            output_name = 'training_flow_changes.png'
    else:
        title = f'Training Flow - {log_dir} (Individual Runs)'
        if output_name is None:
            output_name = f'training_flow_{log_dir}.png'
    
    # Create 5 separate plots, one for each run
    num_runs_to_plot = min(num_runs, len(all_runs))
    fig, axes = plt.subplots(num_runs_to_plot, 1, figsize=(14, 5*num_runs_to_plot))
    if num_runs_to_plot == 1:
        axes = [axes]
    
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.995)
    
    # Plot each run separately
    for idx, run_num in enumerate(sorted(all_runs.keys())[:num_runs_to_plot]):
        ax = axes[idx]
        data = all_runs[run_num]
        
        # Sort by epoch to ensure correct order
        sorted_indices = sorted(range(len(data['epochs'])), key=lambda i: data['epochs'][i])
        epochs = [data['epochs'][i] for i in sorted_indices]
        train_loss = [data['train_loss'][i] for i in sorted_indices]
        train_acc = [data['train_acc'][i] for i in sorted_indices]
        val_acc = [data['val_acc'][i] for i in sorted_indices]
        
        # Plot training loss on left y-axis
        ax1 = ax
        line1 = ax1.plot(epochs, train_loss, 'b-o', label='Training Loss', linewidth=2.5, markersize=8)
        ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Training Loss', color='b', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', labelsize=12)
        ax1.tick_params(axis='y', labelcolor='b', labelsize=12)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Plot training and validation accuracy on right y-axis
        ax2 = ax1.twinx()
        line2 = ax2.plot(epochs, train_acc, 'g-s', label='Training Accuracy', linewidth=2.5, markersize=8)
        line3 = ax2.plot(epochs, val_acc, 'r-^', label='Validation Accuracy', linewidth=2.5, markersize=8)
        ax2.set_ylabel('Accuracy', color='black', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='black', labelsize=12)
        
        # Combine legends
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='best', fontsize=12, framealpha=0.9)
        
        ax1.set_title(f'Run {run_num} - Training Metrics Over Epochs', fontsize=15, fontweight='bold', pad=10)
        
        # Set x-axis to show all epochs
        ax1.set_xticks(epochs)
        ax1.set_xlim(-0.5, max(epochs) + 0.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"Plot saved as '{output_name}'")
    plt.close()  # Close instead of show to avoid blocking

if __name__ == '__main__':
    # Generate plots for both directories
    print("=" * 60)
    print("Generating plot for logs_speed_baseline...")
    print("=" * 60)
    plot_training_flow('logs_speed_baseline', num_runs=5, output_name='training_flow_baseline.png')
    
    print("\n" + "=" * 60)
    print("Generating plot for logs_speed_changes...")
    print("=" * 60)
    plot_training_flow('logs_speed_changes', num_runs=5, output_name='training_flow_changes.png')
    
    print("\n" + "=" * 60)
    print("Done! Generated both plots.")
    print("=" * 60)
