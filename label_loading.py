import numpy as np
import pandas as pd
import random


def load_training_labels(train_df, ext_df, cfg, random_state=cfg.seed):
    """
    Compare training labels with external pseudolabels and filter based on confidence threshold.
    """
    # Create lookup dictionaries to avoid repeated dataframe filtering
    filename_to_primary = {}
    filename_to_secondary = {}
    
    for _, row in train_df.iterrows():
        filename = row['filename'].split('/')[-1].split('.')[0]
        filename_to_primary[filename] = row['primary_label']
        filename_to_secondary[filename] = row['secondary_labels']
    
    return_list = []
    
    for _, ext_row in tqdm(ext_df.iterrows(), desc="Filtering training data", total=len(ext_df)):
        row_parts = ext_row["row_id"].split("_")
        samplename = row_parts[0]
        timestamp = row_parts[1]
        
        # Normalize predictions in one step
        predictions = ext_row[1:]
        predictions = predictions / predictions.sum()
        
        # Get predictions above threshold with a single operation
        passed_preds = predictions[predictions > cfg.train_label_confidence]
        if passed_preds.empty:
            continue
        
        # Sort by confidence in descending order
        passed_preds = passed_preds.sort_values(ascending=False)
        
        # Get training data info via lookup
        if samplename not in filename_to_primary:
            continue
        
        primary_label = filename_to_primary[samplename]
        secondary_labels = filename_to_secondary[samplename]
        secondaries_filtered = []
        
        # Process secondary labels efficiently
        if secondary_labels not in [[''], None, np.nan, [], "['']", "[]"]:
            if isinstance(secondary_labels, str):
                try:
                    secondary_labels = eval(secondary_labels)
                except:
                    secondary_labels = []
                    
            sec_array = np.array(secondary_labels, dtype=str)
            # Find intersection of secondary labels and passed predictions
            secondaries_filtered = [label for label in passed_preds.index if label in sec_array]
        
        # Determine final primary label using the same logic as before
        if primary_label in passed_preds.index:
            if primary_label != passed_preds.index[0]:
                if secondaries_filtered:
                    primary_label = secondaries_filtered[0]
                    secondaries_filtered = secondaries_filtered[1:]
                else:
                    primary_label = passed_preds.index[0]
        else:
            if secondaries_filtered:
                primary_label = secondaries_filtered[0]
                secondaries_filtered = secondaries_filtered[1:]
            else:
                continue
        
        return_list.append({
            "samplename": samplename + "_" + timestamp,
            "primary_label": primary_label,
            "secondary_labels": str(secondaries_filtered),
            "filename": samplename + ".ogg",
            "filepath": os.path.join(cfg.train_datadir, samplename + ".ogg"),
            "timestamp": timestamp,
        })
    result_df = pd.DataFrame(return_list)

    if len(result_df) > cfg.max_train_samples:
        if cfg.stratified_train_samples:
            # Perform stratified sampling based on primary_label
            print(f"Performing stratified sampling to get a maximum of {cfg.max_train_samples} samples")
            
            # Group by primary_label and calculate proportions
            label_counts = result_df['primary_label'].value_counts()
            label_proportions = label_counts / len(result_df)
            
            # Calculate how many samples to take from each group
            samples_per_label = {label: max(1, int(cfg.max_train_samples * prop)) for label, prop in label_proportions.items()}
            
            # Sample from each group
            sampled_dfs = []
            for label, count in samples_per_label.items():
                group_df = result_df[result_df['primary_label'] == label]
                if len(group_df) > count:
                    sampled_dfs.append(group_df.sample(count, random_state=random_state))
                else:
                    sampled_dfs.append(group_df)
            
            # Combine sampled groups
            result_df = pd.concat(sampled_dfs).reset_index(drop=True)
        else:
            # Simple random sampling
            print(f"Randomly sampling {cfg.max_train_samples} samples")
            result_df = result_df.sample(cfg.max_train_samples, random_state=random_state).reset_index(drop=True)
    
    print(f"Returning {len(result_df)} processed pseudolabels")
    return result_df


def load_pseudolabels(df, cfg, seed=None):
    """
    Load pseudolabels and sample to ensure balanced class representation.
    Returns a maximum of cfg.max_pseudolabels samples with balanced label distribution.
    """
    random_seed = seed if seed is not None else cfg.seed
    np.random.seed(random_seed)
    
    if cfg.debug:
        df = df.sample(min(1000, len(df)), random_state=random_seed).reset_index(drop=True)
    
    label_cols = df.columns[1:]
    
    # Step 1: Find samples where at least one prediction passes the confidence threshold
    # Calculate row sums first to filter out rows with no valid predictions
    df_filtered = df.copy()
    row_sums = df_filtered[label_cols].sum(axis=1)
    df_filtered = df_filtered[row_sums > 0].reset_index(drop=True)
    
    # Normalize probabilities to sum to 1 for each row
    if cfg.normalize_labels:
        df_filtered[label_cols] = df_filtered[label_cols].div(df_filtered[label_cols].sum(axis=1), axis=0)
    
    # Create mask for values that pass threshold
    mask = df_filtered[label_cols] >= cfg.pseudolabel_confidence_threshold
    
    # Get indices and values of samples with predictions above threshold using the mask
    valid_samples = {}  # {row_id: [labels sorted by confidence]}
    
    for idx, row in df_filtered.iterrows():
        row_id = row['row_id']
        # Use the mask to get valid labels for this row
        valid_mask = mask.iloc[idx]
        valid_label_cols = label_cols[valid_mask]
        
        if len(valid_label_cols) > 0:
            # Get confidences for valid labels and sort them
            confidences = row[valid_label_cols]
            # Create (confidence, label) pairs and sort by confidence
            label_conf_pairs = [(label, conf) for label, conf in zip(valid_label_cols, confidences)]
            label_conf_pairs.sort(key=lambda x: x[1], reverse=True)
            # Extract just the sorted labels
            valid_samples[row_id] = [label for label, _ in label_conf_pairs]
    
    print(f"Found {len(valid_samples)} samples with confidence > {cfg.pseudolabel_confidence_threshold}")
    
    # Step 2: Apply stratified sampling if enabled
    if cfg.stratified_pseudolabels and len(valid_samples) > cfg.max_pseudolabels:
        print("Stratified sampling of pseudolabels")
        
        # Create label-to-samples mapping
        label_to_samples = {}
        for sample_id, labels in valid_samples.items():
            for label in labels:
                if label not in label_to_samples:
                    label_to_samples[label] = []
                label_to_samples[label].append(sample_id)
        
        # Calculate target samples per label for balanced distribution
        num_unique_labels = len(label_to_samples)
        target_per_label = max(1, int(cfg.max_pseudolabels / num_unique_labels))
        
        print(f"Found {num_unique_labels} unique bird species in pseudolabels")
        print(f"Target ~{target_per_label} samples per species to stay under {cfg.max_pseudolabels} total")
        
        # Shuffle labels for randomness across folds
        shuffled_labels = list(label_to_samples.keys())
        random.seed(random_seed)
        random.shuffle(shuffled_labels)
        
        # Sample entries for each label
        selected_samples = set()
        
        for label in shuffled_labels:
            samples = label_to_samples[label]
            # Shuffle samples with deterministic but different seed per label
            samples_seed = random_seed + hash(label) % 10000
            random.seed(samples_seed)
            random.shuffle(samples)
            
            # Take at most target_per_label samples
            selected = samples[:target_per_label]
            selected_samples.update(selected)
            
            # Stop if we've reached the max pseudolabels
            if len(selected_samples) >= cfg.max_pseudolabels:
                break
        
        # Filter valid_samples to only include selected samples
        valid_samples = {k: v for k, v in valid_samples.items() if k in selected_samples}
        print(f"Sampled {len(valid_samples)} pseudolabeled samples")
    else:
        print(f"Using randomly selected {cfg.max_pseudolabels} pseudolabels that pass confidence threshold")
    
    # Step 3: Convert to training-compatible DataFrame format
    # First, create a dataframe just with the row_ids we need
    rows_to_keep = random.sample(list(valid_samples.keys()), min(cfg.max_pseudolabels, len(valid_samples)))

    label_df = pd.DataFrame({
        'samplename': rows_to_keep,
        'filename': [f"{'_'.join(x.split('_')[:3])}.ogg" for x in rows_to_keep],
        'timestamp': [int(x.split('_')[3]) for x in rows_to_keep],
        'primary_label': [valid_samples[x][0] for x in rows_to_keep],
        'secondary_labels': [valid_samples[x][1:] for x in rows_to_keep]
    })
    
    # Add filepath column 
    label_df['filepath'] = label_df['filename'].apply(lambda x: os.path.join(cfg.train_soundscapes, x))
    
    return label_df


def load_soft_pseudolabels(df, cfg, seed=None):
    """
    Load soft pseudolabels.
    """
    # Can use different seeds e.g. for different folds
    random_seed = seed if seed is not None else cfg.seed
    np.random.seed(random_seed)

    print(f"Found {len(df)} samples in the pseudolabels")
    if cfg.debug:
        df = df.sample(min(1000, len(df)), random_state=random_seed).reset_index(drop=True)
    
    label_col = df.columns[1:]
    
    # Normalize probabilities to sum to 1 for each row
    if cfg.normalize_labels:
        df[label_col] = df[label_col].div(df[label_col].sum(axis=1), axis=0)
        
    # Sample to restrict to max_pseudolabels
    df = df.sample(min(cfg.max_pseudolabels, len(df)), random_state=random_seed).reset_index(drop=True)
    row_ids = df["row_id"]

    # Convert to training-compatible DataFrame format
    label_df = pd.DataFrame({
        'samplename': row_ids,
        'filename': [f"{'_'.join(x.split('_')[:3])}.ogg" for x in row_ids],
        'primary_label' : ["soft"] * len(row_ids),
        'secondary_labels': [df.loc[df["row_id"]==x].to_numpy()[0, 1:].astype(np.float32) for x in row_ids]
    })
    label_df['filepath'] = label_df['filename'].apply(lambda x: os.path.join(cfg.train_soundscapes, x))
    print(f"Loaded {len(label_df)} pseudolabels")

    return label_df