#!/usr/bin/env python3
"""
Script to parse training logs and create graphs for training and validation loss.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

def parse_log_file(filepath):
    """Parse log file to extract training and validation losses, and total training time."""
    train_steps = []
    train_losses = []
    val_steps = []
    val_losses = []
    total_time_ms = None
    
    with open(filepath, 'r') as f:
        for line in f:
            # Match training loss lines: step:X/Y train_loss:Z train_time:Wms
            train_match = re.search(r'step:(\d+)/\d+ train_loss:([\d.]+).*train_time:([\d.]+)ms', line)
            if train_match:
                step = int(train_match.group(1))
                loss = float(train_match.group(2))
                time_ms = float(train_match.group(3))
                train_steps.append(step)
                train_losses.append(loss)
                # Keep track of the latest training time
                if not np.isnan(time_ms):
                    total_time_ms = time_ms
            
            # Match validation loss lines: step:X/Y val_loss:Z train_time:Wms
            val_match = re.search(r'step:(\d+)/\d+ val_loss:([\d.]+).*train_time:([\d.]+)ms', line)
            if val_match:
                step = int(val_match.group(1))
                loss = float(val_match.group(2))
                time_ms = float(val_match.group(3))
                val_steps.append(step)
                val_losses.append(loss)
                # Keep track of the latest training time (this should be the final total time)
                if not np.isnan(time_ms):
                    total_time_ms = time_ms
    
    return train_steps, train_losses, val_steps, val_losses, total_time_ms

def steps_to_epochs(steps, steps_per_epoch=125):
    """Convert steps to epochs. Default steps_per_epoch based on validation frequency."""
    return [s / steps_per_epoch for s in steps]

def format_loss_label(x, p):
    """Format loss labels to show 2 decimal places."""
    return f'{x:.2f}'

def format_time(ms):
    """Convert milliseconds to a readable time format (hours, minutes, seconds)."""
    if ms is None:
        return "N/A"
    total_seconds = ms / 1000.0
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds:.1f}s"
    elif minutes > 0:
        return f"{minutes}m {seconds:.1f}s"
    else:
        return f"{seconds:.1f}s"

def create_plots(swiglu_log, baseline_log, swiglu_output, baseline_output):
    """Create training and validation loss plots for both models."""
    
    # Parse both log files
    print(f"Parsing {swiglu_log}...")
    swiglu_train_steps, swiglu_train_losses, swiglu_val_steps, swiglu_val_losses, swiglu_total_time = parse_log_file(swiglu_log)
    print(f"  Found {len(swiglu_train_steps)} training loss entries and {len(swiglu_val_steps)} validation loss entries")
    print(f"  Total training time: {format_time(swiglu_total_time)}")
    
    print(f"Parsing {baseline_log}...")
    baseline_train_steps, baseline_train_losses, baseline_val_steps, baseline_val_losses, baseline_total_time = parse_log_file(baseline_log)
    print(f"  Found {len(baseline_train_steps)} training loss entries and {len(baseline_val_steps)} validation loss entries")
    print(f"  Total training time: {format_time(baseline_total_time)}")
    
    # Estimate steps per epoch based on validation frequency (validation happens every 125 steps)
    # This is a reasonable estimate for language model training
    steps_per_epoch = 125
    
    # Convert training steps to epochs
    swiglu_train_epochs = steps_to_epochs(swiglu_train_steps, steps_per_epoch)
    baseline_train_epochs = steps_to_epochs(baseline_train_steps, steps_per_epoch)
    
    # For SwiGLU - Filter validation loss to only points with loss < 4
    swiglu_val_steps_filtered = [s for s, l in zip(swiglu_val_steps, swiglu_val_losses) if l < 4]
    swiglu_val_losses_filtered = [l for l in swiglu_val_losses if l < 4]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Training loss over epochs - full training behavior
    swiglu_time_str = format_time(swiglu_total_time)
    ax1.plot(swiglu_train_epochs, swiglu_train_losses, 'b-', alpha=0.6, linewidth=0.5, label='Training Loss')
    ax1.set_yscale('log')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss (log scale)', fontsize=12)
    ax1.set_title(f'SwiGLU: Training Loss (Full Training Behavior)\nTotal Time: {swiglu_time_str}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.yaxis.set_major_formatter(FuncFormatter(format_loss_label))
    ax1.legend()
    
    # Plot 2: Validation loss over steps - only points with loss < 4
    ax2.plot(swiglu_val_steps_filtered, swiglu_val_losses_filtered, 'r-', marker='o', markersize=4, linewidth=1.5, label='Validation Loss')
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title(f'SwiGLU: Validation Loss (Loss < 4)\nTotal Time: {swiglu_time_str}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(FuncFormatter(format_loss_label))
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(swiglu_output, dpi=300, bbox_inches='tight')
    print(f"Saved SwiGLU plots to {swiglu_output}")
    print(f"  Validation points with loss < 4: {len(swiglu_val_steps_filtered)} out of {len(swiglu_val_steps)}")
    plt.close()
    
    # For Baseline GPT - Filter validation loss to only points with loss < 4
    baseline_val_steps_filtered = [s for s, l in zip(baseline_val_steps, baseline_val_losses) if l < 4]
    baseline_val_losses_filtered = [l for l in baseline_val_losses if l < 4]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Training loss over epochs - full training behavior
    baseline_time_str = format_time(baseline_total_time)
    ax1.plot(baseline_train_epochs, baseline_train_losses, 'b-', alpha=0.6, linewidth=0.5, label='Training Loss')
    ax1.set_yscale('log')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss (log scale)', fontsize=12)
    ax1.set_title(f'Baseline GPT: Training Loss (Full Training Behavior)\nTotal Time: {baseline_time_str}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.yaxis.set_major_formatter(FuncFormatter(format_loss_label))
    ax1.legend()
    
    # Plot 2: Validation loss over steps - only points with loss < 4
    ax2.plot(baseline_val_steps_filtered, baseline_val_losses_filtered, 'r-', marker='o', markersize=4, linewidth=1.5, label='Validation Loss')
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title(f'Baseline GPT: Validation Loss (Loss < 4)\nTotal Time: {baseline_time_str}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(FuncFormatter(format_loss_label))
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(baseline_output, dpi=300, bbox_inches='tight')
    print(f"Saved Baseline GPT plots to {baseline_output}")
    print(f"  Validation points with loss < 4: {len(baseline_val_steps_filtered)} out of {len(baseline_val_steps)}")
    plt.close()

if __name__ == "__main__":
    swiglu_log = "swiglu_logs_2/f256f8b3-ddd8-4c19-85e8-3ee0c770d036.txt"
    baseline_log = "baseline_gpt_logs/2f3a008f-d1c4-46eb-9d19-bf7997f6488c.txt"
    swiglu_output = "swiglu_training.png"
    baseline_output = "baseline_gpt_training.png"
    
    create_plots(swiglu_log, baseline_log, swiglu_output, baseline_output)

