import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

def safe_savefig(filepath, dpi=120):
    """Fast & safe figure saver on Windows."""
    try:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception:
                pass
        plt.savefig(filepath, dpi=dpi)
        return filepath
    except OSError:
        base, ext = os.path.splitext(filepath)
        alt_path = f"{base}_latest{ext}"
        plt.savefig(alt_path, dpi=dpi)
        return alt_path

def popup_gui_window(image_paths):
    """Displays a single clean GUI window with tabs for all 3 output charts. No extra gallery app opening."""
    try:
        import tkinter as tk
        from tkinter import ttk
        from PIL import Image, ImageTk

        valid_paths = [p for p in image_paths if os.path.exists(p)]
        if not valid_paths:
            return

        root = tk.Tk()
        root.title("Standard vs ReAct Prompting Benchmark Results")
        root.geometry("1000x750+100+50")

        # Set up tabbed Notebook container
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook.Tab', font=('Segoe UI', 10, 'bold'), padding=[12, 6])

        notebook = ttk.Notebook(root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        photo_refs = []  # Explicit hard references to prevent Garbage Collection

        tab_titles = [
            "Comprehensive Dashboard",
            "Accuracy Comparison",
            "Latency vs Reasoning"
        ]

        for idx, path in enumerate(valid_paths):
            frame = ttk.Frame(notebook)
            tab_name = tab_titles[idx] if idx < len(tab_titles) else os.path.basename(path)
            notebook.add(frame, text=f"  {tab_name}  ")

            img = Image.open(path)
            img.thumbnail((960, 640), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            photo_refs.append(photo)

            lbl = tk.Label(frame, image=photo, bg="white")
            lbl.image = photo  # Prevent Garbage Collector deletion
            lbl.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        root.photo_refs = photo_refs

        # Bring window to front cleanly
        root.attributes('-topmost', True)
        root.update()
        root.attributes('-topmost', False)
        root.focus_force()

        root.mainloop()
    except Exception as e:
        print(f"Error opening GUI window: {e}")

def visualize_results(csv_path: str = None):
    if csv_path is None:
        csv_path = os.path.abspath(os.path.join("results", "experiment_log.csv"))
    
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found. Please run main.py first.")
        return

    df = pd.read_csv(csv_path)

    # Fast theme rendering
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.size': 10})

    results_dir = os.path.abspath("results")
    os.makedirs(results_dir, exist_ok=True)
    generated_charts = []

    # 1. Comprehensive 4-Panel Performance Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Standard vs ReAct Prompting Benchmark Dashboard", fontsize=14, fontweight="bold", y=0.98)

    # Subplot 1: Accuracy Comparison
    ax1 = sns.barplot(ax=axes[0, 0], x="technique", y="accuracy", data=df, hue="technique", errorbar=None, palette="viridis", legend=False)
    axes[0, 0].set_title("1. Mean Accuracy", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylim(0, 1.15)
    for c in ax1.containers:
        ax1.bar_label(c, fmt='%.2f', padding=3)

    # Subplot 2: Reasoning Quality Comparison
    ax2 = sns.barplot(ax=axes[0, 1], x="technique", y="reasoning_quality", data=df, hue="technique", errorbar=None, palette="magma", legend=False)
    axes[0, 1].set_title("2. Mean Reasoning Quality (1-5)", fontsize=11, fontweight="bold")
    axes[0, 1].set_ylim(0, 5.5)
    for c in ax2.containers:
        ax2.bar_label(c, fmt='%.2f', padding=3)

    # Subplot 3: Latency vs Total Tokens
    sns.scatterplot(
        ax=axes[1, 0],
        data=df,
        x="total_tokens",
        y="latency_sec",
        hue="technique",
        style="technique",
        s=100,
        palette="viridis"
    )
    axes[1, 0].set_title("3. Latency vs. Token Consumption", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Total Tokens Used")
    axes[1, 0].set_ylabel("Latency (Seconds)")

    # Subplot 4: Hallucination Rate
    df["hallucination_numeric"] = df["hallucination_present"].astype(int)
    ax4 = sns.barplot(ax=axes[1, 1], x="technique", y="hallucination_numeric", data=df, hue="technique", errorbar=None, palette="rocket", legend=False)
    axes[1, 1].set_title("4. Hallucination Rate", fontsize=11, fontweight="bold")
    axes[1, 1].set_ylabel("Hallucination Proportion")
    axes[1, 1].set_ylim(0, 1.15)
    for c in ax4.containers:
        ax4.bar_label(c, fmt='%.2f', padding=3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    chart_dash_target = os.path.join(results_dir, "comprehensive_dashboard.png")
    chart_dash_saved = safe_savefig(chart_dash_target, dpi=120)
    plt.close()
    generated_charts.append(chart_dash_saved)
    print(f"Saved: {chart_dash_saved}")

    # 2. Bar Chart: Average Accuracy by Technique
    plt.figure(figsize=(7, 5))
    ax = sns.barplot(
        x="technique", 
        y="accuracy", 
        data=df, 
        hue="technique", 
        errorbar=None, 
        palette="viridis", 
        legend=False
    )
    plt.title("Average Accuracy: Standard vs ReAct Prompting", fontsize=12, fontweight="bold", pad=12)
    plt.ylabel("Accuracy (0.0 to 1.0)", fontsize=10)
    plt.xlabel("Prompting Technique", fontsize=10)
    plt.ylim(0, 1.15)
    
    for i in ax.containers:
        ax.bar_label(i, padding=3, fmt='%.2f', fontweight="bold")
        
    plt.tight_layout()
    chart1_target = os.path.join(results_dir, "accuracy_comparison.png")
    chart1_saved = safe_savefig(chart1_target, dpi=120)
    plt.close()
    generated_charts.append(chart1_saved)
    print(f"Saved: {chart1_saved}")

    # 3. Scatter Plot: Latency vs Reasoning Quality
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df, 
        x="latency_sec", 
        y="reasoning_quality", 
        hue="technique", 
        style="technique",
        s=140, 
        alpha=0.85,
        palette="viridis"
    )
    
    plt.title("Latency vs. Reasoning Quality", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Latency (Seconds)", fontsize=10)
    plt.ylabel("Reasoning Quality Score (1-5)", fontsize=10)
    plt.yticks(range(1, 6))
    plt.tight_layout()
    chart2_target = os.path.join(results_dir, "latency_vs_reasoning.png")
    chart2_saved = safe_savefig(chart2_target, dpi=120)
    plt.close()
    generated_charts.append(chart2_saved)
    print(f"Saved: {chart2_saved}")

    # Display clean single-window GUI on screen
    print("\nDisplaying benchmark GUI window on your screen...")
    popup_gui_window(generated_charts)

if __name__ == "__main__":
    visualize_results()
